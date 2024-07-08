from trainer import Trainer
from model import GPT

model = GPT.from_config("config/news_model.yml")
model.to("cuda")

trainer = Trainer.from_config("config/news_trainer.yml")
trainer.train(model)