import torch
import torch.nn as nn
import math
from torch.nn import functional as F

def timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000) / (half_dim - 1))).to(timesteps.device) # (d_model // 2)
    emb = timesteps * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb 

class ViTDecoder(nn.Module):
    def __init__(self, d_model=256, img_size=(48, 64), num_heads=8, num_layers=8, num_feats=8):
        """
        Enhanced Vision Transformer Decoder with U-Net style skip connections.
        
        Args:
            d_model: Feature dimension.
            img_size: Tuple representing input image size.
            num_heads: Number of attention heads in the Transformer.
            num_layers: Number of Transformer encoder layers.
        """
        super().__init__()
        self.d_model = d_model
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, d_model//4, kernel_size=3, stride=1, padding=1),  # (B, 1, 48, 64) -> (B, d_model//4, 48, 64)
            nn.SiLU(),
            nn.Conv2d(d_model//4, d_model//4, kernel_size=5, stride=1, padding=2),  # (B, d_model//4, 48, 64) -> (B, d_model//4, 48, 64)
            nn.SiLU(),
            nn.Conv2d(d_model//4, d_model//2, kernel_size=5, stride=1, padding=2),  # (B, d_model//4, 48, 64) -> (B, d_model//2, 48, 64)
            nn.SiLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(d_model//2, d_model//2, kernel_size=3, stride=2, padding=1),  # (B, d_model//2, 48, 64) -> (B, d_model//2, 24, 32)
            nn.SiLU(),
            nn.Conv2d(d_model//2, d_model//2, kernel_size=5, stride=1, padding=2),  # (B, d_model//2, 24, 32) -> (B, d_model//2, 24, 32)
            nn.SiLU(),
            nn.Conv2d(d_model//2, d_model, kernel_size=5, stride=1, padding=2),  # (B, d_model//2, 24, 32) -> (B, d_model, 24, 32)
            nn.SiLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),  # (B, d_model//2, 24, 32) -> (B, d_model, 24, 32)
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=5, stride=1, padding=2),  # (B, d_model, 24, 32) -> (B, d_model, 12, 16)
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=5, stride=1, padding=2),  # (B, d_model, 12, 16) -> (B, d_model, 12, 16)
            nn.SiLU()
        )
        
        # interpolate to 24, 32
        self.conv4 = nn.Sequential(
            nn.Conv2d(d_model, d_model//2, kernel_size=3, stride=1, padding=1),  # (B, d_model, 24, 32) -> (B, d_model//2, 24, 32)
            nn.SiLU(),
            nn.Conv2d(d_model//2, d_model//2, kernel_size=5, stride=1, padding=2),  # (B, d_model//2, 24, 32) -> (B, d_model//2, 24, 32)
            nn.SiLU(),
            nn.Conv2d(d_model//2, d_model//2, kernel_size=5, stride=1, padding=2),  # (B, d_model//2, 24, 32) -> (B, d_model//2, 24, 32)
            nn.SiLU()
        )
        # interpolate to 48, 64
        self.conv5 = nn.Sequential(
            nn.Conv2d(d_model//2, d_model//2, kernel_size=3, stride=1, padding=1),  # (B, d_model//2, 48, 64) -> (B, d_model//4, 48, 64)
            nn.SiLU(),
            nn.Conv2d(d_model//2, d_model//4, kernel_size=5, stride=1, padding=2),  # (B, d_model//4, 48, 64) -> (B, d_model//4, 48, 64)
            nn.SiLU(),
            nn.Conv2d(d_model//4, 1, kernel_size=5, stride=1, padding=2),  # (B, d_model//4, 48, 64) -> (B, 1, 48, 64)
            # nn.Tanh()
        )
        
        
        # Transformer encoder: Process the flattened low-resolution feature map.
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                       dim_feedforward=d_model*4, dropout=0.1, activation='gelu', batch_first=True),
            num_layers=num_layers
        )
        
        if any(size <= 0 or size % 4 for size in img_size):
            raise ValueError("ViTDecoder image dimensions must be positive multiples of four")
        self.num_tokens = (img_size[0] // 4) * (img_size[1] // 4)
        self.num_feats = num_feats
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens + self.num_feats, d_model) * 0.02) # 
        
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        '''
        self.feat_embed = nn.Sequential(
            nn.Linear(8, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        '''
        
        self.feat_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        ) if num_feats else None

    
    def forward(self, x: torch.Tensor, f: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for EnhancedViTDecoder.
        
        Args:
            x: Input image tensor of shape (B, 1, 48, 64). [-1, 1]
            f: Sensor tensor of shape (B, num_feats), or None for an unconditional model.
            t: Time tensor of shape (B, 1).
        
        Returns:
            noise_map: Predicted noise map of the same shape as x.
        """
        B = x.shape[0]
        
        feat1 = self.conv1(x)# (B, 1, 48, 64) -> (B, d_model//2, 48, 64)  
        feat2 = self.conv2(feat1)   # (B, d_model//4, 48, 64) -> (B, d_model, 24, 32)
        feat3 = self.conv3(feat2)   # (B, d_model, 24, 32) -> (B, d_model, 12, 16)
        
        B, C, H, W = feat3.shape
        feat3_flat = feat3.flatten(2).transpose(1, 2)  

        t_embed = timestep_embedding(t, self.d_model)
        t_embed = self.time_embed(t_embed).unsqueeze(1) # (B, 1, d_model)
          
        # f_embed = self.feat_embed(f).unsqueeze(1) # (B, 8) -> (B, 1, d_model)

        tokens = feat3_flat
        if self.num_feats:
            if f is None or f.shape != (B, self.num_feats):
                raise ValueError(f"Expected {self.num_feats} sensor values per sample")
            features = self.feat_mlp(f.unsqueeze(-1).float())
            tokens = torch.cat([tokens, features], dim=1)
        elif f is not None:
            raise ValueError("The unconditional model does not take sensor values")
        # tokens = tokens + self.pos_embed + t_embed + f_embed
        tokens = tokens + self.pos_embed + t_embed # no feature embedding
        tokens = self.transformer_encoder(tokens)  

        tokens = tokens[:, :H * W, :]  

        feat3_enhanced = tokens.transpose(1, 2).reshape(B, C, H, W)  # (B, d_model, 12, 16)
        feat3_enhanced = feat3_enhanced + feat3
        
        feat3_enhanced = F.interpolate(feat3_enhanced, scale_factor=2) # (B, d_model, 12, 16) -> (B, d_model, 24, 32)
        feat3_enhanced = feat3_enhanced + feat2
        up1 = self.conv4(feat3_enhanced)# (B, d_model, 24, 32) -> (B, d_model//2, 24, 32)
        up2 = F.interpolate(up1, scale_factor=2) # (B, d_model//2, 24, 32) -> (B, d_model//2, 48, 64)      
        up2 = up2 + feat1
        
        noise_map = self.conv5(up2) # (B, d_model//4, 48, 64) -> (B, 1, 48, 64)
        
        return noise_map
