# gpt-lora-from-scratch

A from-scratch implementation of a GPT plus LoRA fine-tuning, built as a learning roadmap with heavily annotated notebooks.

## What's inside

- `notebooks/` — step-by-step walkthrough (start at `01`)
- `bigram.py` — baseline bigram model
- `gpt.py` — tiny GPT training script
- `lora.py` / `gpt_lora.py` — LoRA implementation and fine-tuning
- `sample.py` — compare base GPT vs. LoRA-adapted output

## Data

Training texts are not included. Download links are in the notebooks:

- GPT corpus: tinyshakespeare
- LoRA corpus: A Study in Scarlet (Project Gutenberg)

Desclaimer: Project Gutenberg hosts works that are public domain **in the US**.
Check local copyright law before downloading.

## Usage

\`\`\`bash
python gpt.py              # train the base GPT
python gpt_lora.py     # fine-tune with LoRA
python sample.py       # generate and compare
\`\`\`

## Acknowledgements

Based on Andrej Karpathy's GPT from Scratch lectures and Sebastian Raschka's DoRA from Scratch blog.

## License

MIT
