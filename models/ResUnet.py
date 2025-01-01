import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Identity mapping (shortcut connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # Add the residual (skip) connection
        out = self.relu(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.res_block(x)
        skip = x  # skip connection for the decoder
        x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResidualBlock(out_channels * 2, out_channels)  # Concatenate skip connection

    def forward(self, x, skip):
        x = self.upconv(x)  # Perform upsampling
        if x.shape[2:] != skip.shape[2:]:  # Check for mismatched spatial dimensions
            skip = torch.nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel axis
        x = self.res_block(x)  # Pass through residual block
        return x

    #def forward(self, x, skip):
    #    x = self.upconv(x)
    #    x = torch.cat([x, skip], dim=1)  # Concatenate along channel axis
    #    x = self.res_block(x)
    #    return x

class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnet, self).__init__()

        # Encoder
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

        # Decoder
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        # Final layer to output the result
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1, skip1 = self.encoder1(x)
        enc2, skip2 = self.encoder2(enc1)
        enc3, skip3 = self.encoder3(enc2)
        enc4, skip4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder path
        dec4 = self.decoder4(bottleneck, skip4)
        dec3 = self.decoder3(dec4, skip3)
        dec2 = self.decoder2(dec3, skip2)
        dec1 = self.decoder1(dec2, skip1)

        # Final convolution
        out = self.final_conv(dec1)
        print(out.shape)
        return out

#TODO : code tests for the model 
