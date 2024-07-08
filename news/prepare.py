import sentencepiece as spm
from datasets import load_dataset
import numpy as np
import os

ds = load_dataset("habanoz/eco-news-tr")

train_data = ds['train']
validation_data = ds['validation']

sp = spm.SentencePieceProcessor(model_file="eco_news_tr.model")

train_data_docs_encoded = [[sp.bos_id()]+sp.encode(doc['text'])+[sp.eos_id()] for doc in train_data]
val_data_docs_encoded = [[sp.bos_id()]+sp.encode(doc['text'])+[sp.eos_id()] for doc in validation_data]

train_ids = [code for codes in train_data_docs_encoded for code in codes]
val_ids = [code for codes in val_data_docs_encoded for code in codes]

print(f"Number of Training tokens: {len(train_ids)}")
print(f"Number of Validation tokens: {len(val_ids)}")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
