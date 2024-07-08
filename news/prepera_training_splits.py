import os
import numpy as np
import json
import random
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="eco_news.model")

with open("clean_docs.json", encoding="utf-8") as f:
    docs = json.load(f)

n = len(docs)
keys = list(docs.keys())

random.seed(123)
random.shuffle(keys)

train_data_keys = keys[:int(n*0.9)]
val_data_keys = keys[int(n*0.9):]

train_data_docs = [docs[id] for id in train_data_keys]
val_data_docs = [docs[id] for id in val_data_keys]

train_data_docs_encoded = [[sp.bos_id()]+sp.encode(doc)+[sp.eos_id()] for doc in train_data_docs]
val_data_docs_encoded = [[sp.bos_id()]+sp.encode(doc)+[sp.eos_id()] for doc in val_data_docs]

train_ids = [code for codes in train_data_docs_encoded for code in codes]
val_ids = [code for codes in val_data_docs_encoded for code in codes]

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))