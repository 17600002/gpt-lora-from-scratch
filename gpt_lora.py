import torch
import torch.nn as nn

from gpt import GPTLanguageModel, generate, get_device, get_batch, estimate_loss
from lora import freeze_base_model, add_lora, count_params

import time

# --------------------------------
# Hyperparemeters
# --------------------------------

LORA_SAVE_NAME = 'lorat1'                   # query, value
MODEL_SAVE_NAME = 'gpt_shakes_best'
PATH = f'checkpoints/{MODEL_SAVE_NAME}.pt'

rank = 8
alpha = 16

max_iters = 5000
eval_interval = 250
eval_iters = 20
learning_rate = 1e-3

device = get_device()

# --------------------------------
# Load Pre-trained Model
# --------------------------------

checkpoints = torch.load(PATH, map_location=device, weights_only=False)
config = checkpoints['config']

model = GPTLanguageModel(
    vocab_size=config['vocab_size'],
    n_embd=config['n_embd'],
    n_head=config['n_head'],
    block_size=config['block_size'],
    n_layer=config['n_layer'],
    dropout=config['dropout'],
).to(device)
print(f'Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters')

model.load_state_dict(checkpoints['model_state_dict'])
stoi = config['stoi']
itos = config['itos']
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(itos[i] for i in l)

# --------------------------------
# Prepare LoRA Dataset
# --------------------------------

original_chars = sorted(list(stoi))

with open('data/astudyinscarlet.txt', 'r', encoding='UTF-8') as f:
    raw_text = f.read()
raw_chars = sorted(list(set(raw_text)))

# Replace equivalents
sync = {
    '\u2014': '-',                         # – —  en/em dash -> hyphen
    '\u2018': "'", '\u2019': "'",          # ‘ ’  curly single quotes

    '\u00e8': 'e',   # è  -> e  (grave accent stripped)
    '\u00e9': 'e',   # é  -> e  (acute accent stripped)
    '\u00f1': 'n',   # ñ  -> n  (tilde stripped)
}

for new, old in sync.items():
    raw_text = raw_text.replace(new, old)

# Drop every character not in the original vocab
text = ''.join(ch for ch in raw_text if ch in original_chars)

# Split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# --------------------------------
# Add LoRA to Model
# --------------------------------

# First freeze the pre-trained model
freeze_base_model(model)

# Then add LoRA layers
# add_lora_all(model, rank, alpha)                                      # add to every layer
add_lora(model, rank, alpha, targets=('query', 'value'))                # tier 1
# add_lora(model, rank, alpha, targets=('query','key','value','proj'))    # tier 2

model.to(device)

count_params(model)

# --------------------------------
# Training
# --------------------------------

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad), lr=learning_rate
)

model.train()

start = time.time()

print('Training starts: ')
t0 = time.time()

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters-1:
        losses = estimate_loss(model, train_data, val_data,
                                eval_iters, config['batch_size'], config['block_size'], device)
        print(f"Step {step}: train {losses['train']:.4f} | val {losses['val']:.4f}")

    xb, yb = get_batch(train_data, config['batch_size'], config['block_size'], device)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"  step {step}  {(time.time()-t0)/(step+1):.3f}s/it", flush=True)

elapsed = time.time() - start
print(f'Training took {elapsed//60:.0f}m {elapsed%60:.0f}s')

# Save LoRA
lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k}
torch.save(lora_state, f'checkpoints/{LORA_SAVE_NAME}_{MODEL_SAVE_NAME}.pt')
print(len(lora_state))

print(generate(
        model, encode, decode, device=device,
        prompt='\n', max_new_tokens=500))