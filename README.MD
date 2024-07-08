# HaberGPT

HaberGPT is a simple framework influenced greatly by the works of Andrej Karpathy, namely mingpt and nanogpt.

HaberGPT is built to train a tiny GPT2 language model using economy news in Turkish language. It shows that it is possible to train a 10M language model that can generate coherent Turkish text using limited scale e.g. hardware and training tokens.

HaberGPT is not a research project. It is a proof of concept and a reproduction study. 

## Model

HaberGPT is a GPT2 model with a 13.77M parameters. It has 6 layers, 6 attention heads and 384 embeddings. Sequence length is 512 tokens.

## Tokenizer

HaberGPT trains its own tokenizer using the BPE algorithm implementation of the sentencepiece library. Tokenizer is trained using the training corpus which is Turkish language only. Vocabulary size is selected to be 8192. A larger vocabulary may help increase sequence length, but it is not studied. `<s>` and `</s>` special tokens are used to indicate document boundaries.  

## Dataset

Influenced by the Tiny Stories paper, HaberGPT is trained on a specialized topic that includes a rather limited vocabulary and a typical structure. Unlike Tiny Stories paper which was trained on synthetic data, Ha
berGPT is trained on economy news articles that are crawled from web. 

Dataset contains 8K documents which can be downloaded from huggingface at repository location `habanoz/eco-news-tr`. 

Training split has 4.5M tokens, and validation split has 0.5M tokens. 


## Training

Training script requires a CUDA hardware with bf16 floating point support. For further speed-up, mixed precision mode and fused AdamW optimizer is used. It can greatly benefit from flash attention on a supporting hardware e.g. Ampere or recent. My old 1070 card does not have flash attention support and training took almost 4 hours. A recent GPU should have significantly less time to complete training. 

Gradient accumulation is not supported for code simplicity but can be added easily. In fact I would recommend you to refer to nanoGPT work of Andrej Karpathy.

Batch size was 32, which can be increased further if GPU memory is not an issue. 

The model is trained for 5000 steps, for a total of `32 x 512 x 5000=82M` tokens. 

According to Chinchilla scaling laws (Training Compute-Optimal Large Language Models), for a compute-optimal training, the ratio of tokens to model parameters is approximately **20:1**. As a result our model, which has **13M** parameters and which should be trained using `13M x 20 = 260M` tokens, is clearly in under-trained regime. 


## Instructions

Train the tokenizer:

```bash
python news/train_tokenizer.py
```

Prepare training/validation data splits.

```bash
python news/prepare.py
```

Train the model. Edit `config/news_model.yml` and `config/news_trainer.yml` configuration files to customize the model or training.

```bash
python train_haber_gpt.py
```

## Experimental Generations

This is not a research study, thus it is evaluated very lightly.


## References

1- [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)

2- [TinyStories: How Small Can Language Models Be and Still SpeakCoherent English?](https://arxiv.org/pdf/2305.07759)

3- [NanoGPT](https://github.com/karpathy/nanoGPT)

4- [eco-news-tr dataset](https://huggingface.co/datasets/habanoz/eco-news-tr)

5- [Training Code(This repository)](https://github.com/habanoz/haber-gpt)