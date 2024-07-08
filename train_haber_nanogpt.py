from trainer import Trainer
from nanogpt import GPT, GPTConfig

model = GPT(GPTConfig())
model.to("cuda")

trainer = Trainer.from_config("config/news_trainer.yml")
trainer.train(model)