import torch
import torch.nn as nn
from .blocks import DoubleConv, CrossAttentionBlock

class UNet2_5D(nn.Module):
    """
    2.5D Dual-Encoder Cross-Attention U-Net for Prostate Segmentation.
    Expects inputs of shape (B, 3, H, W) where 3 represents the stacked Z-axis slices.
    """
    def __init__(self, in_channels=3, n_classes=1, features=[64, 128, 256, 512], use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # T2 Encoder (Anatomy - Provides Query & Skips).
        self.t2_enc1 = DoubleConv(in_channels, features[0])
        self.t2_enc2 = DoubleConv(features[0], features[1])
        self.t2_enc3 = DoubleConv(features[1], features[2])
        self.t2_enc4 = DoubleConv(features[2], features[3])
        
        # ADC Encoder (Function - Provides Key & Value).
        self.adc_enc1 = DoubleConv(in_channels, features[0])
        self.adc_enc2 = DoubleConv(features[0], features[1])
        self.adc_enc3 = DoubleConv(features[1], features[2])
        self.adc_enc4 = DoubleConv(features[2], features[3])
        
        # Bottleneck & Cross Attention.
        self.t2_bottleneck = DoubleConv(features[3], features[3] * 2)
        self.adc_bottleneck = DoubleConv(features[3], features[3] * 2)

        if self.use_attention:
            # Input to attention is 1024 (features[3] * 2) channels.
            self.cross_attention = CrossAttentionBlock(in_channels=features[3] * 2)
        else:
            # Fallback(no attention): linearly compresses the concatenated bottlenecks (2048) back to 1024.
            # 1 * 1 Convolution.
            self.fallback_fusion = nn.Conv2d(features[3] * 4, features[3] * 2, kernel_size=1)
        
        # DECODER (Driven exclusively by T2 skip connections).
        # dec is skip connection concatenation -> DoubleConv.
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3]) 
        
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])
        
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])
        
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])
        
        # Final 1x1 Convolution to output the binary mask.
        self.head = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, t2, adc):
        # T2 Encoder Pass (Save skip connections).
        t2_s1 = self.t2_enc1(t2)
        t2_s2 = self.t2_enc2(self.pool(t2_s1)) # Applying max pooling.
        t2_s3 = self.t2_enc3(self.pool(t2_s2))
        t2_s4 = self.t2_enc4(self.pool(t2_s3))
        
        # ADC Encoder Pass (No skip connections saved).
        adc_s1 = self.adc_enc1(adc)
        adc_s2 = self.adc_enc2(self.pool(adc_s1))
        adc_s3 = self.adc_enc3(self.pool(adc_s2))
        adc_s4 = self.adc_enc4(self.pool(adc_s3))
        
        # Bottleneck.
        t2_b = self.t2_bottleneck(self.pool(t2_s4))
        adc_b = self.adc_bottleneck(self.pool(adc_s4))

        # Cross-Attention.
        if self.use_attention:
            fused_b, attention_map = self.cross_attention(t2_features=t2_b, adc_features=adc_b)
        else:
            # Concatenate the 1024-channel bottlenecks into a 2048-channel tensor.
            concat_b = torch.cat([t2_b, adc_b], dim=1)
            # Fuse back to 1024 channels via 1x1 convolution.
            fused_b = self.fallback_fusion(concat_b)
            attention_map = None
        
        # Decoder Pass.
        # We concatenate (dim=1) the Up-convolution with the corresponding T2 skip connection.
        d4 = self.up4(fused_b)
        d4 = torch.cat([d4, t2_s4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, t2_s3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, t2_s2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, t2_s1], dim=1)
        d1 = self.dec1(d1)
        
        logits = self.head(d1)
        
        return logits, attention_map