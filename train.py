import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
# preprocessing



# model

class cross_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # conv block
        
        self.resconv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose3d(128, out_channels, kernel_size=2, stride=2)

    def forward(self, x): # x: [B, C, D, H, W]
        
