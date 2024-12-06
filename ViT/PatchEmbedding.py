import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

device = torch.device('cuda')

class PatchEmbedding(nn.Module):
    def __init__(self, dim, patch_size, input_size):
        super().__init__()
        self.seq_length = int((((input_size - patch_size)/patch_size + 1) ** 2) + 1)
        self.tokenize = Tokenize(patch_size, 3, dim)
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn((1, self.dim, 1))).to(device)
        self.positions = nn.Parameter(torch.randn((1, self.dim, self.seq_length))).to(device)

    def add_classtoken(self, input_sequence):
        cls_token = self.cls_token.expand(input_sequence.size(0), -1, -1)
        output_seq = torch.cat([cls_token, input_sequence], dim=-1)
        return output_seq
    
    def position_embedding(self, input_seq):
        output_seq = input_seq + self.positions
        return output_seq
    
    def forward(self, x):
        token = self.tokenize(x)
        output = self.add_classtoken(token)
        output = self.position_embedding(output)
        return output

class Tokenize(nn.Module):
    def __init__(self, patch_size, input_channel, out_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, out_dim, patch_size, stride=patch_size, padding=0),
            nn.BatchNorm2d(out_dim),
            Rearrange('b c h w -> b c (h w)')
        )
  
    def forward(self, x):
        x = self.layer(x)
        return x


        