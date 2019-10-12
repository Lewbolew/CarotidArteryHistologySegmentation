import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import UNet
from models.shape_net import ShapeUNet
BN_EPS = 1e-4


class SH_UNet(nn.Module):

    def __init__(self, path_to_shape_net_weights='', n_classes=15):
        super(SH_UNet, self).__init__()

        self.unet = UNet((3, 512,512))
        self.shapeUNet = ShapeUNet((15, 512,512))
        self.softmax = nn.Softmax(dim=1)
        if path_to_shape_net_weights:
            self.shapeUNet.load_state_dict(torch.load(path_to_shape_net_weights))

    def forward(self, x, only_encode=False):
        if only_encode:
            _, encoded_mask = self.shapeUNet(x)
            return encoded_mask

        unet_prediction = self.unet(x)
        softmax_unet_prediction = self.softmax(unet_prediction)#.detach()
        shape_net_final_prediction, shape_net_encoded_prediction = self.shapeUNet(softmax_unet_prediction)

        return unet_prediction, shape_net_encoded_prediction, shape_net_final_prediction

