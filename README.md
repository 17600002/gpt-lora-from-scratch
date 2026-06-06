# GPT from Scratch + LoRA

A from-scratch implementation of a GPT plus LoRA fine-tuning, built as a learning roadmap with heavily annotated notebooks.

## Summary

- A 10M param tiny GPT trained on TinyShakespeare -> babbles Shakespeare-ish prose
- LoRA adapted model on A Study in Scarlet -> now Victorian-like
- Base model ~ 130 MB | LoRA adapter ~ 2 MB | Trainable params ~ 3%

## Sample Generation

### GPT-Shakespeare
```plain text
Shut that they warrant you?

LADY CAPULET:
It is myself; lay bear you.
Hear not myself? you disminate peace:
The eye would crimine of courtesy complaints
Or, stubbing thee, reward, richsiving and mad
Is church shapes, that they should bestow'd, I say those tongue
That takes the dest.

```

### BASE + LoRA-Sherlock
```plain text
Kept you put your pleasure agreet.

If I have said yielded, unclined with his delight; and there was
counterness to mine opinion her which your life to his familiar.
You must not give her head, I have it saw my heart as her is
followed that you did else? he said, soft.
```

## What's Inside

### Notebooks
Builds the model step by step. Detailed dual notation comparing math and tensor shapes side by side.  
| Notebook | Content |
|---|---|
| 01_bigram-dev | Bigram LM |
| 02_attention_math| bag of words, single-head self-attention in detail|
| sample | compare output of the trained pure GPT and when added LoRA |

In progress:
| Notebook | Content |
|---|---|
| 03_gpt-dev| a toy GPT decoder |
| 04_lora-test| LoRA on a toy feed forward network (conceptual)|
| 05_gpt_lora-dev | LoRA fine-tuning on toy GPT |

### Scripts
| Scripts | Content |
|---|---|
| bigram.py | baseline bigram model |
| gpt.py | tiny GPT training script |
| lora.py | LoRA implementation |
| gpt_lora.py | LoRA fine-tuning |
| sample.py | compare base GPT vs. LoRA-adapted output |

## Use

```bash
python gpt.py              # train base GPT
python gpt_lora.py         # fine-tune with LoRA
python sample.py           # generate and compare
```



## Data

Training texts are not included. Download links are also in the notebooks:

- GPT corpus: Tiny Shakespeare
- LoRA corpus: A Study in Scarlet (Project Gutenberg)

Download Tiny Shakespear with:

```bash
wget -O data/tinyshakespear.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
Download A Study in Scarlet with


```bash
wget -O data/astudyinscarlet.txt https://www.gutenberg.org/cache/epub/244/pg244.txt
```

Disclaimer: Project Gutenberg hosts works that are public domain **in the US**.
Check local copyright law before downloading.



## Acknowledgements

Based on [Andrej Karpathy's GPT from Scratch lectures](https://youtu.be/kCc8FmEb1nY?si=2gBghBYD0H0BfjpZ) and [Sebastian Raschka's DoRA from Scratch blog](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch).

Transformer paper:  [Attention Is All You Need](https://arxiv.org/abs/2106.09685)

LoRA paper: 
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Next Steps
Update notebooks and add KV cache

## License

MIT
