import torch

from gpt import GPTLanguageModel, generate, get_device
from lora import add_lora

# -----------------------------------
# Config
# -----------------------------------

CHECKPOINTS_PATH = 'checkpoints'
BASE_PATH = f'{CHECKPOINTS_PATH}/gpt_shakes_best.pt'
LORA_PATH = f'{CHECKPOINTS_PATH}/lorat1_gpt_shakes_best.pt'

PROMPT = '\n'
MAX_NEW_TOKENS = 500

device = get_device()

# -----------------------------------
# Functions
# -----------------------------------

def load_base(config, device):

    model = GPTLanguageModel(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        dropout=config['dropout'],
    ).to(device)
    model.load_state_dict(checkpoints['model_state_dict'])
    return model



# -----------------------------------
# Load vocab
# -----------------------------------

checkpoints = torch.load(BASE_PATH, map_location=device, weights_only=False)
config = checkpoints['config']

stoi = config['stoi']
itos = config['itos']

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(itos[i] for i in l)

# -----------------------------------
# Load Model
# -----------------------------------

# 1. Pure GPT
GPT_model = load_base(config, device)
GPT_model.eval()

# 2. GPT + LoRA
lora_model = load_base(config, device)
add_lora(lora_model, rank=8, alpha=16, targets=('query', 'value')) # need to match training
lora_model.to(device)

lora_state = torch.load(LORA_PATH, map_location=device)
lora_model.load_state_dict(lora_state, strict=False)
lora_model.eval()

# -----------------------------------
# Sample and compare!
# -----------------------------------

print('[BASE (Tiny Shakespeare)]')
print(generate(GPT_model, encode, decode, device, 
            prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS))

print('\n[BASE + LoRA (Conan Doyle)]')
print(generate(lora_model, encode, decode, device, 
            prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS))
