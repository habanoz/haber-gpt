from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import numpy as np
import os
import math
import time
import datetime
from collections import defaultdict

DEVICE="cuda"

@dataclass
class TrainerConfig:
    seq_length: int = 1024
    batch_size: int = 64
    data_dir: str = None
    warmup_iters: int = 100
    learning_rate: float = 1e-4
    lr_decay_iters: int = 5000 
    max_iters: int = 5000
    min_lr: float = 1e-6
    weight_decay: float = None
    beta1: float = 0.9
    beta2: float= 0.95
    compile: bool=False
    decay_lr: bool = True
    seed: int = 145
    log_interval: int = 10
    eval_interval: int = 250
    eval_iters: int = 200
    out_dir:str = "out"
    wandb_log: bool = False
    wandb_project: str = "GPT Training"
    wandb_run_name: str = "run1"
    grad_norm_clip: float = 1.0

class Trainer:
    def __init__(self, config:TrainerConfig) -> None:
        self.config = config

        assert torch.cuda.is_available(), "Cuda is not available. This training script requires an NVIDIA GPU!"
        assert torch.cuda.is_bf16_supported(), "BF16 is not supported. This training script requires BF16 support!"

        self.ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def from_config(config_file):
        import yaml

        with open(config_file) as f:
            try:
                doc = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)
        
        config = TrainerConfig(**doc)
        return Trainer(config)
            

    def get_batch(self, split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(self.config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        ix = torch.randint(len(data) - self.config.seq_length, (self.config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.config.seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.config.seq_length]).astype(np.int64)) for i in ix])
        
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)

        return x, y

    def update_lr(self, it, optimizer):
        if not self.config.decay_lr:
            return self.config.learning_rate
        
        lr = self.get_lr(it)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
        
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def _configure_optimizers(self, model):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        betas = (self.config.beta1, self.config.beta2)
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=betas, fused=True)
        
        return optimizer
    
    def _init_logging(self):
        if self.config.wandb_log:
            import wandb
            # wandb.require("core")
            wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name, config=asdict(self.config))
    
    @torch.no_grad()
    def estimate_loss(self, model:nn.Module):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def train(self, model:nn.Module):
        # assert model.device == DEVICE, f"Only CUDA device is supported for training. Model is in :{model.device}"
        assert self.config.seq_length == model.config.seq_length, f"Sequence length for model and trainer is not equal {self.config.seq_length}!={model.config.seq_length}"

        torch.manual_seed(self.config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        if self.config.compile:
            print("compiling the model...")
            model = torch.compile(model)
            print("compiling the model done!")
            
        self._init_logging()

        optimizer = self._configure_optimizers(model)

        # start training
        model.train()
    
        # fetch the very first batch
        X, Y = self.get_batch('train')
        
        start_iter = 0
        best_val_loss = 1e+9
        running_fwd_bwd_tokens_per_sec = 0
        running_iter_time = 0
        for iter in range(start_iter, self.config.max_iters+1):
            
            # determine current lr rate and update params if needed
            lr = self.update_lr(iter, optimizer)
            
            if iter % self.config.eval_interval == 0:
                self.do_eval(model, optimizer, best_val_loss, running_fwd_bwd_tokens_per_sec, running_iter_time, iter, lr)
            
            # forward the model
            t0 = time.time()
            with self.ctx:
                _, loss = model(X, Y)
            
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = self.get_batch('train')

            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
            optimizer.step()
            
            t1 = time.time()
            if iter % self.config.log_interval == 0:
                loss_item = loss.item() # GPU/CPU sync point

                dt = t1 - t0
                
                fwd_bwd_tokens = self.config.batch_size * self.config.seq_length
                fwd_bwd_tokens_per_sec = fwd_bwd_tokens * (1.0 / dt)
                
                if running_fwd_bwd_tokens_per_sec==0:
                    running_fwd_bwd_tokens_per_sec = fwd_bwd_tokens_per_sec
                    running_iter_time = dt
                else:
                    running_fwd_bwd_tokens_per_sec = 0.9*running_fwd_bwd_tokens_per_sec + 0.1*fwd_bwd_tokens_per_sec
                    running_iter_time = 0.9* running_iter_time + 0.1 * dt

                print(f"iter {iter}: loss {loss_item:.4f}, run_iter_time {running_iter_time*1000:.2f}ms, fb_toks/sec {fwd_bwd_tokens_per_sec:.2f}, run_fb_toks/sec {running_fwd_bwd_tokens_per_sec:.2f}")


    def do_eval(self, model, optimizer, best_val_loss, running_fwd_bwd_tokens_per_sec, time_per_iter, iter, lr):
        t0 = time.time()
        losses = self.estimate_loss(model)
                
        if self.config.wandb_log:
            import wandb
            wandb.log({
                        "iter": iter,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "tokens/sec": running_fwd_bwd_tokens_per_sec,
                    })
                
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter > 0:
                checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter,
                            'best_val_loss': best_val_loss,
                        }
                print(f"saving checkpoint to {self.config.out_dir}")
                if not os.path.exists(self.config.out_dir):
                    os.makedirs(self.config.out_dir)
                torch.save(checkpoint, os.path.join(self.config.out_dir, 'ckpt.pt'))
                
        t1 = time.time()
        dt = t1 - t0

        if running_fwd_bwd_tokens_per_sec> 0:
            # remaining training iterations
            rem_iters = self.config.max_iters - iter
            rem_fwd_bwd_time = rem_iters * time_per_iter
            # remaining evaluation iterations
            rem_evals = rem_iters // self.config.eval_interval
            rem_evals_time = dt * rem_evals
            # remaining time
            eta = rem_fwd_bwd_time + rem_evals_time 
        else:
            eta = 0.0

        print(f"Eval iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, ETA {datetime.timedelta(seconds=eta)}")