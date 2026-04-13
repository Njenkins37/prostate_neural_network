import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Standard UNet building block: Two sequential Convolution -> BatchNorm -> ReLU layers.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CrossAttentionBlock(nn.Module):
    """
    Spatial Cross-Attention bottleneck.
    T2 provides the Query (Anatomy). ADC provides the Key and Value (Function).
    """
    def __init__(self, in_channels):
        super().__init__()
        # Reduce channels by a factor of 8 for Q and K to save memory (standard practice).
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable parameter to scale the attention output before adding the residual.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, t2_features, adc_features):
        # Batch size, Channels, Height, Width.
        B, C, H, W = t2_features.size()

        # Generate Q, K, V.
        # Flatten spatial dimensions (H, W) into a single sequence (N).
        # Attention only understands 1D, so .view() changes it to (B,C,N).
        proj_query = self.query_conv(t2_features).view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C')
        proj_key = self.key_conv(adc_features).view(B, -1, H * W)                      # (B, C', N)
        proj_value = self.value_conv(adc_features).view(B, -1, H * W)                  # (B, C, N)

        # Calculate Attention Scores (Q * K^T).
        energy = torch.bmm(proj_query, proj_key)  # (B, N, N), batch matrix multiplication.
        attention_map = F.softmax(energy, dim=-1)

        # Apply Attention to ADC Values.
        out = torch.bmm(proj_value, attention_map.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, H, W) 

        # Residual Connection: Add original T2 anatomy back into the attended functional features.
        fused_out = self.gamma * out + t2_features
        
        return fused_out, attention_map