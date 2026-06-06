import torch
import torch.nn as nn
from torch.nn import functional as F

#-------------------------
# Hyperparameters
#-------------------------

INPUT_FILE = 'data/tinyshakespear.txt'

max_iters = 5000
eval_interval = 300
eval_iters = 200

learning_rate = 1e-3

batch_size = 4
block_size = 8

# Device agnostic
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f'Using device: {device}')


#-------------------------
# Models and functions
#-------------------------

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # Embedding table shape (V, d_model) - both `vocab_size` here
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):           # input batch indices `idx` 
                                                    # shape (B, T)

        # Simplifies embedding for logits here
        logits = self.token_embedding_table(idx)    # logits shape (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape                  # reshape for F.cross_entropy
            logits = logits.view(B*T, C)            # (B, T, C) -> (B*T, C)
            targets = targets.view(B*T)             # (B, T) -> (B*T)

            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Forward pass
            logits, loss = self(idx)                    
            logits = logits[:, -1, :]                   # (B, T, C) -> (B, C)
                                                        # only the last token matters
            # Softmax probability
            probs = F.softmax(logits, dim=-1)      

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)     
                                                        
            # Append sampled index 
            idx = torch.concat((idx, idx_next), dim=1)          # (B, T+1)       

        return idx

def get_batch(split, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    xb = torch.stack([data[i : i+block_size] for i in ix]).to(device)
    yb = torch.stack([data[i+1 : i+block_size+1] for i in ix]).to(device)
    
    return xb, yb

@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#-------------------------
# Load Data
#-------------------------

with open(INPUT_FILE, 'r', encoding='UTF-8') as f:
    text = f.read()

# Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)           # V for number of tokens

# Token <-> Index Mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]          # Take a str, output a list of int
decode = lambda l: ''.join([itos[i] for i in l])   # Take a list of int, output a str

# Load data to torch
data = torch.tensor(encode(text), dtype=torch.long)

# Train-val split
n_train = int(0.9 * len(data))
train_data = data[:n_train]
val_data = data[n_train:]

#-------------------
# Initialize model
#-------------------

model = BigramLanguageModel(vocab_size)
m = model.to(device)


#-------------------
# Training
#-------------------


# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}| val loss {losses['val']:.4f}")

    # Sample batch
    xb, yb = get_batch('train', batch_size, block_size)

    # Forward pass
    logits, loss = model(xb, yb)

    # Evaluate the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate from the model
seed_token = torch.zeros((1, 1), dtype=torch.long).to(device)
max_new_tokens = 300
generated_tokens = decode(m.generate(seed_token, max_new_tokens)[0].tolist())
print(generated_tokens)
    


