import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
import torch.nn.functional as F

device = torch.device('cuda')
class Encoder(nn.Module):
    def __init__(self, emb_dim=128, patch_size=32, input_size=64, num_heads=8, mlp_num=8):
        super().__init__()
        self.emb_dim = emb_dim
        self.token_length = int((((input_size - patch_size)/patch_size + 1) ** 2) + 1)
        self.mlp_size = mlp_num
        self.layerNorm = nn.LayerNorm(emb_dim) # LN
        self.multihead_attention = MHA(emb_dim=emb_dim, num_heads=num_heads)
        self.MLP = nn.Sequential(
            nn.Linear(self.emb_dim, self.mlp_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.mlp_size, self.emb_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def LN(self, input_tensor):
        normalized_tensor = self.layerNorm(input_tensor.transpose(1, 2))
        normalized_tensor = normalized_tensor.transpose(1, 2)
        
        return normalized_tensor

    def forward(self, input_tensor):
        residual_1 = input_tensor
        normalized_tensor = self.LN(input_tensor)
        after_MHA, attention_map = self.multihead_attention(normalized_tensor)
        residual_2 = (after_MHA + residual_1)
        before_MLP = self.LN(residual_2) 
        after_MLP = self.MLP(before_MLP.transpose(1,2)).transpose(1,2)
        output = (after_MLP + residual_2)
        return output, attention_map

class MHA(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.attn_dim = emb_dim // num_heads
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.weight_Q = nn.Linear(emb_dim, emb_dim)
        self.weight_K = nn.Linear(emb_dim, emb_dim)
        self.weight_V = nn.Linear(emb_dim, emb_dim)

        self.out_projection = nn.Linear(emb_dim, emb_dim)
        self.rearrange = Rearrange("b s (h a) -> b s h a", h=num_heads, a=self.attn_dim)
        self.reshape = Rearrange("b s h a -> b s (h a)", h=num_heads, a=self.attn_dim)
        
    def split_heads(self, x):
        x = self.rearrange(x)
        return x.permute(0, 2, 1, 3)

    def self_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attn_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, input_token):
        input_token = input_token.transpose(1,2)
        Q = self.split_heads(self.weight_Q(input_token))  # (batch_size, num_heads, seq_len, attn_dim)
        K = self.split_heads(self.weight_K(input_token))
        V = self.split_heads(self.weight_V(input_token))
        attention_output = self.self_attention(Q, K, V)  # (batch_size, num_heads, seq_len, attn_dim)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, attn_dim)
        attention_output = self.reshape(attention_output)
        output = self.out_projection(attention_output)  # (batch_size, seq_len, emb_dim)
        output = output.transpose(1, 2)
        return output, attention_output.transpose(1,2)

class MLP_Head(nn.Module):
    def __init__(self, emb_dim=128, input_size=224, patch_size=32):
        super(MLP_Head, self).__init__()
        self.token_length = int((((input_size - patch_size)/patch_size + 1) ** 2) + 1)

        self.head = nn.Linear(emb_dim, 10)

    def forward(self, x):
        x = self.head(x)
        
        return x
    