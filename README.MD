# HaberGPT

HaberGPT is a simple framework influenced greatly by the works of Andrej Karpathy, namely mingpt and nanogpt. Please note that it is not a copy of the mentioned works, it is rather my implementation for my use case. 

HaberGPT is built to train a tiny GPT2 language model using economy news in Turkish language. It shows that it is possible to train a 10M language model to generate coherent Turkish text using a limited compute e.g. hardware and training tokens.

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

You can see WANDB training logs:
https://wandb.ai/huseyinabanozis/GPT%20Training?nw=nwuserhuseyinabanozis


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

Edit the prompt in the `generate.py` and produce completions using following command.
```bash
python generate.py
```

## Experimental Generations

This is not a research study, thus it is evaluated very lightly.

**Prompt:** `# Borsa güne yükselişle başladı`
```
# Borsa güne yükselişle başladı ## Özet Borsa İstanbul'da BIST 100 endeksi, güne yüzde 0,53 artışla 3.507,09 puandan başladı. ## İçerik Açılışta BIST 100 endeksi, önceki kapanışa göre 9,16 puan ve yüzde 0,53 değer kazanarak 4.507,09 puana çıktı. 
Bankacılık endeksi yüzde 0,34 ve holding endeksi yüzde 0,66 artış kaydetti. Sektör endeksleri arasında en fazla kazandıran yüzde 4,96'yla iletişim, en çok kaybettiren yüzde 0,23'le menkul kıymet yatırım ortaklığı oldu. 
BIST 100 endeksi dün günü yüzde 0,63 artışla 4.528,46 puandan tamamlayarak tüm zamanların en yüksek günlük ve haftalık kapanışını gerçekleştirirken, gördüğü en yüksek seviye rekorunu 4.568,98 puana taşıdı. 
Analistler, küresel pay piyasalarında, ABD Merkez Bankası'nın (Fed) perşembe günkü para politikası kararları öncesinde bankacılık sektörüne ilişkin endişelerin risk iştahını azaltantıcı etki incelemesi ve ABD'de açıklanan Üretici Fiyat Endeksi (ÜFE) verilerinin öngörülerin üzerinde artmasına karşın ülkedeki ekonomik görünüme ilişkin endişelerin günden güne yavaş azaldığını ifade ederek, bugün ülkede açıklanacak enflasyon verilerinin yatırımcıların odağında bulunacağını bildirdi. 
Öte yandan, dün Koç Holding, Yapı ve Kredi Bankası'nın Koç Holding, Garanti Bankası'nın düzenlediği 43 milyon lirayla Gözdemirkcell ve Garanti Bankası'na 24,2 milyon lira nominal değerli hisse senedi piyasalarından yüzde 54'ü aştığını bildirdi. 
Bugün yurt içinde arzını beklentilerin üzerinde karlılıkları ile makroekonomik veri takvimi de yatırımcıların odağında yer alacağını aktaran analistler, teknik açıdan BIST 100 endeksinde 4.50 ve 4.750 seviyelerinin direnç, 4.60 ve 4.600 puanın destek konumunda olduğunu kaydetti.
```

Generated text is coherent and reflects the structure present in the training data. However, new lines are not generated. Generation is highly senstive to prompt similarity to the training data.


**Prompt:** `# Borsa güne yükselişle başladı ## Özet Sporda düşüş sürdü.`
```
# Borsa güne yükselişle başladı ## Özet Sporda düşüş sürdü. ## İçerik İSTANBUL **Borsa İstanbul'da **BIST 100 endeksi**, güne yüzde 0,43 artışla 1.200,47 puandan başladı. 
Açılışta **BIST 100 endeksi**, 33,70 puan ve yüzde 0,43 artışla 1.200,47 puana çıktı. Bankacılık endeksi yüzde 0,56 değer kazanırken, holding endeksi ise yüzde 0,66 değer kaybetti. 
Sektör endeksleri arasında en fazla kazandıran yüzde 2,62 ile turizm, en çok gerileyen ise yüzde 1,70 ile menkul kıymet yatırım ortaklığı oldu. 
Dün analistlerin direnç olarak nitelendirdikleri 1.388,85 puana kadar gerileyen BIST 100 endeksi, günü önceki kapanışa göre yüzde 0,57 yükselişle 1.19,06 puandan tamamladı. 
Analistler, bugün yurt içinde veri akışının sakin olduğunu, yurt dışında ise Avro Bölgesi güven endeksi ve ABD'de hizmet sektörü Satın Alma Yöneticileri Endeksi (PMI) verileri ile ABD Merkez Bankası (Fed) Başkanı Janet Yellen'ın konuşmasının takip edileceğini belirtti. 
ABD ve Avrupa'da açıklanacak imalat sanayi Satın Alma Yöneticileri Endeksi (PMI) verisinin de piyasaların yönü üzerinde etkili olduğunu kaydeden analistler, BIST 100 endeksinde 1.200 puan seviyesinin önemli destek konumuna geldiğini bildirdi.
```

This prompt includes a summary section that mentions sports stocks falling. However, generated text does not include this detail. This sample shows that text completion does not correctly account for prompt variations. 

Short comings should be resolved by introducing more compute e.g. parameters and training tokens.

## References

1- [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)

2- [TinyStories: How Small Can Language Models Be and Still SpeakCoherent English?](https://arxiv.org/pdf/2305.07759)

3- [NanoGPT](https://github.com/karpathy/nanoGPT)

4- [eco-news-tr dataset](https://huggingface.co/datasets/habanoz/eco-news-tr)

5- [Training Code(This repository)](https://github.com/habanoz/haber-gpt)
