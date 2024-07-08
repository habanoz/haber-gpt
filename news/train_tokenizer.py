from datasets import load_dataset
import sentencepiece as spm

ds = load_dataset("habanoz/eco-news-tr")

with open("long_text.txt", "w", encoding="utf-8") as f:
    for doc in ds['train']:
        f.write(doc['text'])
    for doc in ds['validation']:
        f.write(doc['text'])

spm.SentencePieceTrainer.Train(input="long_text.txt", model_prefix="eco_news_tr", vocab_size=2**13, model_type="bpe", split_digits=True)