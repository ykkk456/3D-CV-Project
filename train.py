import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
# preprocessing



# model

class cross_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()
        self.linear_e = nn.Linear(in_channels, out_channels)
        self.linear_d1 = nn.Linear(in_channels, out_channels)
        self.linear_d2 = nn.Linear(in_channels, out_channels)
        self.num_heads = num_heads
    def forward(self, e, d): # e: [B, C, D, H, W], d: [B, C, D, H, W]
        # 特征变换，做多头注意力
        B, C, D, H, W = e.shape
        N = D * H * W
        out_dim = self.linear_e.out_features
        head_dim = out_dim // self.num_heads

        # 展平空间维度: (B, C, D, H, W) → (B, N, C)
        e_flat = e.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        d_flat = d.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Q ← linear_e(e), K ← linear_d1(d), V ← linear_d2(d)
        # 投影: (B, N, C) → (B, N, out_dim)
        Q = self.linear_e(e_flat)
        K = self.linear_d1(d_flat)
        V = self.linear_d2(d_flat)

        # 拆多头: (B, N, out_dim) → (B, num_heads, N, head_dim)
        K = K.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        Q = Q.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # 计算注意力权重
        attn = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)  # (B, num_heads, N, N)
        attn = torch.softmax(attn, dim=-1)

        # 加权 V 并合并多头: (B, num_heads, N, head_dim) → (B, N, out_dim)
        out = attn @ V  # (B, num_heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, out_dim)  # (B, N, out_dim)

        # 重塑回原始空间尺寸: (B, N, out_dim) → (B, out_dim, D, H, W)
        out = out.transpose(1, 2).view(B, out_dim, D, H, W)

        return out








class seg_head(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor,
                              mode='trilinear', align_corners=False)
        return x 
    

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # conv block
        self.bn1 = nn.BatchNorm3d(1)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)
        
        
        self.resconv1 = nn.Conv3d(1, 128, kernel_size=1)
        self.resconv2 = nn.Conv3d(128, 256, kernel_size=1)
        self.resconv3 = nn.Conv3d(256, 512, kernel_size=1)
        self.resconv4 = nn.Conv3d(512, 1024, kernel_size=1)
        
        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)

        self.deconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose3d(128, out_channels, kernel_size=2, stride=2)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.seghead1 = seg_head(512, 1, scale_factor=8)
        self.seghead2 = seg_head(256, 1, scale_factor=4)
        self.seghead3 = seg_head(128, 1, scale_factor=2)
        self.seghead4 = seg_head(out_channels, 1, scale_factor=1)

        self.attn1 = cross_attention(in_channels=512, out_channels=512, num_heads=8)
        self.attn2 = cross_attention(in_channels=256, out_channels=256, num_heads=8)
        self.attn3 = cross_attention(in_channels=128, out_channels=128, num_heads=8)
        self.attn4 = cross_attention(in_channels=out_channels, out_channels=out_channels, num_heads=8)

    def forward(self, x): # x: [B, C, D, H, W]
        # encoder
        x = self.bn1(x)
        x1 = self.resconv1(x) + self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x2 = self.bn2(x1)
        x2 = self.resconv2(x2) + self.conv2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x3 = self.bn3(x2)
        x3 = self.resconv3(x3) + self.conv3(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)
        x4 = self.bn4(x3)
        x4 = self.resconv4(x4) + self.conv4(x4)
        x4 = self.relu(x4)
        x4 = self.maxpool(x4)

        # decoder & seghead & attn
        x4 = self.deconv1(x4)
        x4_out = self.seghead1(x4)
        x3 = self.attn1(x4, x3) 
        x3 = self.deconv2(x3)
        x3_out = self.seghead2(x3)
        x2 = self.attn2(x3, x2)
        x2 = self.deconv3(x2)
        x2_out = self.seghead3(x2)
        x1 = self.attn3(x2, x1)
        x1 = self.deconv4(x1)
        x1_out = self.seghead4(x1)



        return x4_out, x3_out, x2_out, x1_out