import torch
from torch.nn import functional as F
from model import GPT
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="eco_news_tr.model")

checkpoint = torch.load("news-out/ckpt.pt")
model = GPT.from_config("config/news_model.yml")
model.load_state_dict(checkpoint['model'])
model.eval()

prompt = torch.tensor([sp.bos_id()]+sp.Encode("# Borsa güne yükselişle başladı")).view(1, -1)

for i in range(512):
    logits,_ = model(prompt)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=1)
    tok = torch.multinomial(probs, 1, replacement=True)

    if tok == sp.eos_id():
        break

    prompt = torch.cat((prompt , tok), dim=-1)

text = sp.Decode(prompt.tolist()[0])

print(text)