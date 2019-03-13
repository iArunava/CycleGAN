import torch
import torch.nn as nn

from BNConv import BNConv

class Discriminator(nn.Module):
    '''
    '''

    def __init__(self, iconv=64, layers=4):
        
        super(Discriminator, self).__init__()
        
        self.net = nn.ModuleList([nn.Conv2d(in_channels=3,
                                              out_channels=iconv,
                                              kernel_size=4,
                                              stride=2,
                                              padding=1,
                                              bias=False),
                                    
                                 nn.LeakyReLU(negative_slope=0.2)
                                ])

        layers -= 1

        for _ in range(layers):
            self.net.append(BNConv(in_channels=iconv,
                                  out_channels=iconv*2,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1,
                                  relu='leaky',
                                  nslope=0.2,
                                  norm='instance')
                            )

            iconv *= 2

        self.net.append(nn.Conv2d(in_channels=iconv,
                                  out_channels=1,
                                  kernel_size=2,
                                  stride=1,
                                  padding=0,
                                  bias=False)


    def forward(self, x):
        
        for layer in self.net:
            x = layer(x)

        return x
