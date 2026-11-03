import torch.nn as nn
from typing import Type
from .se import SE

class MBConvBlock(nn.Module):

    def __init__(self, 
                 dim: int,
             ):
        super().__init__()
        
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        
    def forward(self, x):
        x = x.contiguous()
        x =  x + self.mb_conv(x) # skip connection

        return x

class GCViTPatchEmbed(nn.Module):
    
    """"
        out: (B,C,H,W)
        inp: (B,H,W,D)
    """
    
    def __init__(
                self,
                in_chs: int,
                dim: int,
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                ):
        super().__init__()
        
        """
        Args:
            in_chs: Feature Map Channels
            dim: Embedding Dimension
        """
        
        self.proj = nn.Sequential(
            nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0),
            MBConvBlock(dim)
        )

        self.norm = norm_layer(dim)    
          
    def forward(self, x):
        x = self.proj(x) # (B, in_chs, H, W) -> (B, dim, H, W)
        x = x.permute(0,2,3,1) # (B, H, W, dim)
            
        return self.norm(x)