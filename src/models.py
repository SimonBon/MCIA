import torch
from torch import nn
from mmengine.registry import MODELS

class ConvBlock(nn.Module):
    def __init__(self, in_channels, groups, block_width, drop_prob=0.0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            padding=1,
            groups=groups, 
            bias=False)
        
        self.LN = nn.BatchNorm2d(in_channels)
        
        self.linear1 = nn.Conv2d(
            in_channels, 
            in_channels * block_width, 
            kernel_size=1,
            groups=groups, 
            bias=False)
        
        self.GELU = nn.GELU()
        
        self.linear2 = nn.Conv2d(
            in_channels * block_width, 
            in_channels, 
            kernel_size=1,
            groups=groups, 
            bias=False)
        
        # Optional spatial dropout
        self.drop = nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.LN(x)
        x = self.linear1(x)
        x = self.GELU(x)
        x = self.linear2(x)
        
        x = self.drop(x)
        
        x += identity
        return x

@MODELS.register_module()
class WideModel(nn.Module):
    
    def __init__(self, in_channels, stem_width=16, block_width=4, layer_config=[2, 2], drop_prob=0.05, late_fusion=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.stem_width = stem_width
        self.block_width = block_width
        self.stem_out_channels = in_channels * self.stem_width
        self.layer_config = layer_config
        self.drop_prob=drop_prob
        
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                self.stem_out_channels, 
                kernel_size=3, 
                padding=1, 
                groups=in_channels, 
                bias=False
            ),
            nn.BatchNorm2d(self.stem_out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layers = []
        for config in self.layer_config:
            self.layers.append(self._make_layer(config))
            self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        
        if late_fusion:
            
            self.late_fusion = nn.Conv2d(
                self.stem_out_channels,
                self.stem_out_channels,
                kernel_size=1)
            
        else:
            self.late_fusion = None
            
        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, n_blocks):
        layers = []
        
        for _ in range(n_blocks):
            layers.append(ConvBlock(
                in_channels=self.stem_out_channels, 
                groups=self.in_channels,
                block_width=self.block_width,
                drop_prob=self.drop_prob
            ))
            
        return nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        
        try:
        
            x = self.stem(x)
            x = self.layers(x)
            x = self.avgpool(x)
            
        except Exception as e:
            
            if x.shape[-1]//2**len(self.layer_config) < 1:
                print('!!! ERROR MOST LIKELY DUE TO TOO MANY DOWNSAMPLINGS REDUCE len(layer_configs) !!!')
                
            raise e

        if self.late_fusion is not None:
            x = self.late_fusion(x)

        
        return (x,)


@MODELS.register_module()
class SharedStemModel(nn.Module):
    
    def __init__(self, in_channels, stem_width=16, block_width=4, n_layers=2, late_fusion=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.stem_width = stem_width
        self.block_width = block_width
        self.stem_out_channels = in_channels * stem_width  # Total after concatenation
        
        # Shared stem: 1 -> stem_width for each channel
        self.shared_stem = nn.Sequential(
            nn.Conv2d(1, stem_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True)
        )
        
        # Shared layers: process each channel's stem output
        layers = []
        for _ in range(n_layers):
            layers.append(ConvBlock(
                in_channels=stem_width,  # Input is stem_width per channel
                groups=1,  # Process each channel's features together within the channel
                block_width=block_width
            ))
            
        self.shared_layers = nn.Sequential(*layers)
        
        if late_fusion:
            self.late_fusion = nn.Conv2d(
                self.stem_out_channels,
                self.stem_out_channels,
                kernel_size=1
            )
        else:
            self.late_fusion = None
            
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, *args, **kwargs):
        # x: [B, in_channels, H, W]
        B, C, H, W = x.shape
        
        # Apply shared stem and layers to each channel independently
        feats = []
        for c in range(C):
            # Extract single channel: [B, 1, H, W]
            x_c = x[:, c:c+1, :, :]
            
            # Shared stem: [B, 1, H, W] -> [B, stem_width, H, W]
            x_c = self.shared_stem(x_c)
            
            # Shared layers: [B, stem_width, H, W] -> [B, stem_width, H, W]
            x_c = self.shared_layers(x_c)
            
            feats.append(x_c)
        
        # Stack all channel features: [B, stem_width*C, H, W]
        x = torch.cat(feats, dim=1)
        
        # Late fusion across channels
        if self.late_fusion is not None:
            x = self.late_fusion(x)
        
        # Global pooling: [B, stem_width*C, H, W] -> [B, stem_width*C, 1, 1]
        x = self.avgpool(x)
        return (x,)
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
