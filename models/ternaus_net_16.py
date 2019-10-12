import numpy as np 
import torch
import torchvision
from torch import nn
from torch import cat
from torchvision.models import vgg16
from torch.nn import functional as F
from skimage.transform import resize

class ConvRelu(nn.Module):
    """
    Concatenated Convolution -> Relu
    """
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class InterpolateModule(nn.Module):
    """
    Encapsulated interpolate function just to be able to use it in Sequentail. 
    """
    def __init__(self, size=None, mode='nearest', scale_factor=None, align_corners=False):
        super(InterpolateModule, self).__init__()
        self.interpol = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        return self.interpol(x, size=self.size, scale_factor=self.scale_factor, 
                            mode=self.mode, align_corners=self.align_corners)

class DecoderBlock(nn.Module):
    """
    There is a problem, described in [1]. Used the info for optimal Deconvolution parameters.

    [1] https://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                    ConvRelu(in_channels, middle_channels),
                    nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True) 
                )
        else:
            self.block = nn.Sequential(
                    InterpolateModule(scale_factor=2, mode='bilinear'),
                    ConvRelu(in_channels, middle_channels),
                    ConvRelu(middle_channels, out_channels)
                )
    def forward(self, x):
        return self.block(x)

def small_vgg_block(relu, vgg_layer1, vgg_layer2):
    """
    Small encoder block, extracted from VGG16
    """

    return nn.Sequential(
            vgg_layer1,
            relu,
            vgg_layer2,
            relu
        )

def big_vgg_block(relu, vgg_layer1, vgg_layer2, vgg_layer3):
    """
    Big encoder block, extracted from VGG16
    """
    return nn.Sequential(
            vgg_layer1,
            relu,
            vgg_layer2,
            relu,
            vgg_layer3,
            relu
        )

class TernausNet16(nn.Module):
    """
    Modification of Unet, described in [1].
    Unet with modified encoder part. VGG16 was used as an encoder.

    [1] https://arxiv.org/abs/1801.05746
    """

    def __init__(self, num_classes=1, num_filters=32):
        super(TernausNet16, self).__init__()
        self.num_classes = num_classes

        self.max_pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU(inplace=True)

        self.vgg_encoder = vgg16(pretrained=True).features

        self.encoder_block_1 = small_vgg_block( self.relu,
                                                self.vgg_encoder[0],
                                                self.vgg_encoder[2]
                                                )
        self.encoder_block_2 = small_vgg_block( self.relu,
                                                self.vgg_encoder[5],
                                                self.vgg_encoder[7]
                                                )
        self.encoder_block_3 = big_vgg_block( self.relu,
                                              self.vgg_encoder[10],
                                              self.vgg_encoder[12],
                                              self.vgg_encoder[14]
                                            )
        self.encoder_block_4 = big_vgg_block( self.relu,
                                              self.vgg_encoder[17],
                                              self.vgg_encoder[19],
                                              self.vgg_encoder[21]
                                            )
        self.encoder_block_5 = big_vgg_block( self.relu,
                                              self.vgg_encoder[24],
                                              self.vgg_encoder[26],
                                              self.vgg_encoder[28]
                                            )

        self.bottleneck = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.decoder_block_5 = DecoderBlock(512+num_filters*8, num_filters*8*2, num_filters*8)
        self.decoder_block_4 = DecoderBlock(512+num_filters*8, num_filters*8*2, num_filters*8)
        self.decoder_block_3 = DecoderBlock(256+num_filters*8, num_filters*4*2, num_filters*2)
        self.decoder_block_2 = DecoderBlock(128+num_filters*2, num_filters*2*2, num_filters)
        self.decoder_block_1 = ConvRelu(64+num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoded_1 = self.encoder_block_1(x)
        encoded_2 = self.encoder_block_2(self.max_pool(encoded_1))
        encoded_3 = self.encoder_block_3(self.max_pool(encoded_2))
        encoded_4 = self.encoder_block_4(self.max_pool(encoded_3))
        encoded_5 = self.encoder_block_5(self.max_pool(encoded_4))

        bottleneck = self.bottleneck(self.max_pool(encoded_5))

        decoded_5 = self.decoder_block_5(cat([bottleneck, encoded_5],1))
        decoded_4 = self.decoder_block_4(cat([decoded_5, encoded_4],1))
        decoded_3 = self.decoder_block_3(cat([decoded_4, encoded_3],1))
        decoded_2 = self.decoder_block_2(cat([decoded_3, encoded_2],1))
        decoded_1 = self.decoder_block_1(cat([decoded_2, encoded_1],1))

        out = self.final(decoded_1)

        return out