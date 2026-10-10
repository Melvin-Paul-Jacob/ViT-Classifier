import math
import torch
from torch import nn
from config import vit_config
import torch.nn.functional as F

class VisionEmbedding(nn.Module):
    def __init__(self, config: vit_config):
        super().__init__()
        
        self.config  = config
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=config.num_channels,
                out_channels=config.embd_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
                padding="valid"
            ),
            nn.Flatten(start_dim=2)
        )
        
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, config.embd_dim)), requires_grad=True)
        self.pos_embeddings = nn.Parameter(torch.randn(size=(1, config.num_patches + 1, config.embd_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=config.dropout)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        patch_embd = self.patch_embedding(x).transpose(2,1)
        patch_embd = torch.cat([cls_token, patch_embd], dim=1)
        embd = self.pos_embeddings + patch_embd
        embd = self.dropout(embd)
        return embd

class VisionAttention(nn.Module):
    def __init__(self, config:vit_config):
        super().__init__()
        
        self.embd_dim = config.embd_dim
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        
        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.k_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.v_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        
        query = query.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2)
        key = key.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2)
        value = value.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2)
        
        attn_score = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attn_score = F.softmax(attn_score, dim=-1).to(query.dtype)
        
        attn_out = (attn_score @ value).transpose(1,2)
        attn_out = attn_out.reshape(B, T, C).contiguous()
        attn_out = self.out_proj(attn_out)
        attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)
        
        return attn_out

class VisionMLP(nn.Module):
    def __init__(self, config:vit_config):
        super().__init__()
        
        self.layer1 = nn.Linear(config.embd_dim, 3 * config.embd_dim)
        self.layer2 = nn.Linear(3 * config.embd_dim, config.embd_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = nn.functional.gelu(x, approximate="tanh")
        x = self.layer2(x)
        return x

class VisionEncoderLayer(nn.Module):
    def __init__(self, config: vit_config):
        super().__init__()
        
        self.embd_dim = config.embd_dim
        self.attn = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embd_dim, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embd_dim, eps=config.layer_norm_eps)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class VisionEncoder(nn.Module):
    def __init__(self, config: vit_config):
        super().__init__()
        self.layers = nn.ModuleList([VisionEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config= vit_config):
        super().__init__()
        
        self.embedding = VisionEmbedding(config)
        self.encoder = VisionEncoder(config)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.embd_dim, eps=config.layer_norm_eps),
            nn.Linear(config.embd_dim, config.num_classes)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.mlp_head(x[:, 0, :])
        return x