import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        x = self.scaling * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def freeze_base_model(model):
    for param in model.parameters():
        param.requires_grad = False

def add_lora_all(model, rank, alpha):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, LinearWithLoRA(child, rank, alpha))
        else:
            add_lora(child, rank, alpha)

def add_lora(model, rank, alpha, targets=('query', 'value')):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(model, name, LinearWithLoRA(child, rank, alpha))
        else:
            add_lora(child, rank, alpha, targets)

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} | Total: {total:,}')
    print(f'{100*trainable/total:.2f}%')

