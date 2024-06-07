import torch
import torch.nn as nn

"""
Implemenet ResNetBlock and ResNet.
Add hyperparameters to ResNet model.
Refer to ReseNet paper, freely modify the model, but you must implement the residuals.
"""

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        pass

    def forward(self, x):
        pass


class ResNet(nn.Module):
    def __init__(self, out_class=10):
        super(ResNet, self).__init__()
        # Utilize ResNet Block.
        pass

    def forward(self, x):
        pass
