import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4


class ConvBnRelu2d(nn.Module):
    '''
    Block from witch will consist Encoder and Decoder elementary parts. 
    '''
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class StackEncoder(nn.Module):
    '''
    Block of the Encoder part of the UNet Conv2d -> Conv2d -> MaxPool -> 
    '''
    def __init__(self, in_channels, out_channels, kernel_size=(3,3)):

        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x_small


class StackDecoder(nn.Module):
    '''
    Block of the Decoder part of the Unet 
    Upsampling(Deconv or interpolation) -> Concat(upX and downX) -> ConvBnRelu2d -> ConvBnRelu2d
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, uppsampling_method='conv_transpose'):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.decode = nn.Sequential(
            ConvBnRelu2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            )

        if uppsampling_method == 'conv_transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2,stride=2)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
                )

    def forward(self, x):
        x = self.upsample(x)
        x = self.decode(x)
        return x


class Bridge(nn.Module):
    '''
    The bottleneck in the UNet.
    '''
    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()

        self.bridge = nn.Sequential(
            ConvBnRelu2d(in_channels, out_channels),
            ConvBnRelu2d(out_channels, out_channels)
            )
    def forward(self, x):
        return self.bridge(x)



class ShapeUNet(nn.Module):
    DEPTH = 5
    def __init__(self, in_shape, n_classes=15):
        super(ShapeUNet, self).__init__()

        channels, height, width = in_shape

        # ENCODER PART                                          # Image Size: 
        self.down_1 = StackEncoder(channels, 64, kernel_size=3) # 272
        self.down_2 = StackEncoder(64, 128, kernel_size=3)      # 136
        self.down_3 = StackEncoder(128, 256, kernel_size=3)     # 68
        self.down_4 = StackEncoder(256, 512, kernel_size=3)     # 34
        self.down_5 = StackEncoder(512, 1024, kernel_size=3)    # 17

        # BOTTLENECK
        self.bottleneck = Bridge(1024, 1024)

        # DECODER PART
        self.up_5 = StackDecoder(1024, 512, kernel_size=3)      # 17
        self.up_4 = StackDecoder(512, 256, kernel_size=3)       # 34
        self.up_3 = StackDecoder(256, 128, kernel_size=3)       # 68
        self.up_2 = StackDecoder(128, 64, kernel_size=3)        # 136
        self.up_1 = StackDecoder(64, 32, kernel_size=3)         # 272
        
        self.classify = nn.Conv2d(32, n_classes, kernel_size=1, bias=True)

    def forward(self, x):

        out = self.down_1(x)
        out = self.down_2(out)
        out = self.down_3(out)
        out = self.down_4(out)
        out = self.down_5(out)

        encoded_shape = self.bottleneck(out)

        out = self.up_5(encoded_shape)
        out = self.up_4(out)
        out = self.up_3(out)
        out = self.up_2(out)
        out = self.up_1(out)

        output_shape = self.classify(out)



        return output_shape, encoded_shape
