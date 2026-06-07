import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------------------------
# Default Hyperparameters
# ----------------------------
# Only used when running this script directly

MODEL_SAVE_NAME = 'gpt_shakes'
INPUT_FILE = 'data/tinyshakespear.txt'

BATCH_SIZE = 64
BLOCK_SIZE = 256

MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

LEARNING_RATE = 3e-4

N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
    

# ----------------------------
# Models
# ----------------------------

class Head(nn.Module):
    """ single head self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):                 # (B, T, n_embd) -> (B, T, head_size)
        _, T, _ = x.shape

        q = self.query(x)                 # (B, T, head_size)
        k = self.key(x)                   # (B, T, head_size)

        # Attention score (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # Scale factor d_k**-0.5 = hs**-0.5
        wei = q @ k.transpose(-1, -2) * k.size(-1)**-0.5             # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)                                 # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values (B, T, T) @ (B, T, hs) -> (B, T, hs) 
        v = self.value(x)                 # (B, T, head_size)
        out = wei @ v                     # (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads self-attention """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
            )
        self.proj = nn.Linear(n_embd, n_embd)       # Mix different heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ full transformer block """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()

        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)             # for residual connection
        self.ln2 = nn.LayerNorm(n_embd)             # for residual connection

    def forward(self, x):
        x = x + self.sa(self.ln1(x))                # normalize before attention
        x = x + self.ffwd(self.ln2(x))              # normalize before FFN
        return x
    
class GPTLanguageModel(nn.Module):
    """ bigram language model with self-attention """

    def __init__(self, vocab_size, n_embd, n_head, block_size, dropout, n_layer):
        super().__init__()

        self.block_size = block_size 

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)             # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # input representation = token embedding + positional encoding
        tok_emb = self.token_embedding_table(idx)    # (B, T, n_embd)                     
        pos_emb = self.position_embedding_table(     # (T, n_embd)
            torch.arange(T, device=idx.device)
            )
        x = tok_emb + pos_emb                        # (B, T, n_embd)

        x = self.blocks(x)                           # (B, T, n_embd)
        x = self.ln_f(x)                             # (B, T, n_embd)
        logits = self.lm_head(x)                     # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):         # (B, T) -> (B, T+1)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]      # (B, block_size)

            logits, _ = self(idx_cond)                # (B, block_size, n_embd)
            logits = logits[:, -1, :]                 # (B, n_embd)
            probs = F.softmax(logits, dim=-1)         # (B, n_embd)
            idx_next = torch.multinomial(             # (B, 1)
                probs, num_samples=1
                ) 
            idx = torch.cat((idx, idx_next), dim=-1)  # (B, T+1)
        
        return idx



# ----------------------------
# Helper Functions
# ----------------------------

def get_batch(data, batch_size, block_size, device):
    
    ix = torch.randint((len(data) - block_size), (batch_size,))
    xb = torch.stack([data[i : i+block_size] for i in ix])
    yb = torch.stack([data[i+1 : i+block_size+1] for i in ix])

    xb, yb = xb.to(device), yb.to(device)

    return xb, yb

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):

    out = {}
    model.eval()

    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xe, ye = get_batch(data, batch_size, block_size, device)
            _, loss = model(xe, ye)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

def save_checkpoint(model, optimizer, step, val_loss, config, save_name, save_dir='checkpoints'):

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{save_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss,
        'config': config,
    }, save_path)
    print(f'Checkpoint saved to {save_path}')

def load_data(input_file):

    with open(input_file, 'r', encoding='UTF-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[ch] for ch in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[: n]
    val_data = data[n: ]

    return vocab_size, train_data, val_data, stoi, itos, encode, decode

# ----------------------------
# Training
# ----------------------------

def train(config, device):
    vocab_size, train_data, val_data, stoi, itos, encode, decode = load_data(config['input_file'])

    # Update configuration
    config = {
        **config, 
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos
    }

    # Initialize model
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        dropout=config['dropout'], 
    )
    model.to(device)
    print(f'Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    best_val = float('inf')
    start = time.time()

    # Training loop
    model.eval()

    for step in range(config['max_iters']):
        
        # Visualize the loss
        if step % config['eval_interval'] == 0 or step == config['max_iters']-1:
            losses = estimate_loss(model, train_data, val_data,
                                   config['eval_iters'], config['batch_size'],
                                   config['block_size'], device)
            print(f"Step {step}: train {losses['train']:.4f} | val {losses['val']:.4f}")

            if losses['val'] < best_val:
                best_val = losses['val']
                save_checkpoint(model, optimizer, step, losses['val'], config,
                                save_name=f"{config['model_save_name']}_best")
        
        # Get data
        xb, yb = get_batch(train_data, config['batch_size'], config['block_size'], device)

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start
    print(f'Training took {elapsed//60:.0f}m {elapsed%60:.0f}s')

    save_checkpoint(model, optimizer, step, losses['val'], config,
                    save_name=f"{config['model_save_name']}_final")
    
    return model, encode, decode

def generate(model, encode, decode, device, prompt='\n', max_new_tokens=500):
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = decode(model.generate(context, max_new_tokens)[0].tolist())
    return out
    


# ----------------------------
# Main Script
# ----------------------------

if __name__ == '__main__':

    device = get_device()
    print(f'Using device: {device}')

    config = dict(
        input_file = INPUT_FILE,
        model_save_name = MODEL_SAVE_NAME,
        batch_size = BATCH_SIZE,
        block_size = BLOCK_SIZE,
        max_iters = MAX_ITERS,
        eval_interval = EVAL_INTERVAL,
        eval_iters = EVAL_ITERS,
        learning_rate = LEARNING_RATE,
        n_embd = N_EMBD,
        n_head = N_HEAD,
        n_layer = N_LAYER,
        dropout = DROPOUT,
    )

    # Training
    model, encode, decode = train(config, device)

    # Generation
    print(generate(
        model, encode, decode, device, 
        prompt='To be or not to be', max_new_tokens=500))
