import torch
import torch.nn as nn
from BNConv import BNConv
from BNConvt import BNConvt
from ResidualBlock2L import ResidualBlock2L

class CycleGAN(nn.Module):

    def __init__(self, ic_conv=64):
        '''
        This class defines the network used for CycleGAN.
        This network is solely taken from the Supplementary material provided
        along with the Perceptual Losses paper by Johnson et al.
        '''
        super(CycleGAN, self).__init__()
        
        self.net = nn.Sequential(
                        BNConv(in_channels=3,
                                out_channels=ic_conv,
                                kernel_size=9,
                                padding=4,
                                stride=1,
                                conv_first=True,
                                relu=True),

                        BNConv(in_channels=ic_conv,
                                out_channels=ic_conv*2,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                conv_first=True,
                                relu=True),

                        BNConv(in_channels=ic_conv*2,
                                out_channels=ic_conv*4,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                conv_first=True,
                                relu=True),

                        ResidualBlock2L(ic_conv=ic_conv*4,
                                        oc_conv=ic_conv*4,
                                        downsample=False,
                                        conv_first=True,
                                        relu_after_add=False),

                        ResidualBlock2L(ic_conv=ic_conv*4,
                                        oc_conv=ic_conv*4,
                                        downsample=False,
                                        conv_first=True,
                                        relu_after_add=False),

                        ResidualBlock2L(ic_conv=ic_conv*4,
                                        oc_conv=ic_conv*4,
                                        downsample=False,
                                        conv_first=True,
                                        relu_after_add=False),

                        ResidualBlock2L(ic_conv=ic_conv*4,
                                        oc_conv=ic_conv*4,
                                        downsample=False,
                                        conv_first=True,
                                        relu_after_add=False),

                        ResidualBlock2L(ic_conv=ic_conv*4,
                                        oc_conv=ic_conv*4,
                                        downsample=False,
                                        conv_first=True,
                                        relu_after_add=False),

                        BNConvt(in_channels=ic_conv*4,
                                out_channels=ic_conv*2,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                conv_first=True,
                                relu=True),

                        BNConvt(in_channels=ic_conv*2,
                                out_channels=ic_conv,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                conv_first=True,
                                relu=True),

                        BNConvt(in_channels=ic_conv,
                                out_channels=3,
                                kernel_size=9,
                                padding=4,
                                stride=1,
                                conv_first=True,
                                relu=True)
            )


    def forward(self, x):
        '''
        This function defines the forward pass of the network
        '''

        x = self.net(x)
        return x
