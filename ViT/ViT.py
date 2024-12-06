import torch.nn as nn
from PatchEmbedding import PatchEmbedding
import torch.nn.functional as F
from encoder import Encoder, MLP_Head

class ViT(nn.Module):
    def __init__(self, emb_dim=128, patch_size=32, input_size=224, num_layers=5, num_heads=5, mlp_num=8):
        super(ViT, self).__init__()
        self.emb_dim = emb_dim
        self.patch_emb = PatchEmbedding(emb_dim, patch_size, input_size=input_size).cuda()
        self.MLP_Head = MLP_Head(emb_dim=emb_dim, patch_size=patch_size, input_size=input_size)
        self.encoder_blocks = nn.ModuleList([
            Encoder(emb_dim=emb_dim, patch_size=patch_size, input_size=input_size, num_heads=num_heads, mlp_num=mlp_num)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x = self.patch_emb(x)
        for encoder_block in self.encoder_blocks:
            x, attention_map = encoder_block(x)

        x = x[:, :, 0] # class_token -> MLP_Head
        output = self.MLP_Head(x)
        return output, attention_map
    