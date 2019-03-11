import torch
import torch.nn as nn

class BNConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, bias=False, eps=1e-5, momentum=0.1, conv_first=True, relu='relu', nslope=0.2, norm='batch'):

        super(BNConv, self).__init__()
        
        relu = relu.lower()
        norm = norm.lower()
        
        if norm == 'batch':
            norm = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        elif norm == 'instance':
            norm = nn.InstanceNorm2d(out_channels, eps=eps, momentum=momentum)
        else:
            raise ('Norm value not understood')

        if conv_first:
            self.main = nn.ModuleList([
                            nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=bias),
                            norm
                    ])
            
            if relu == 'relu':
                self.main.append(nn.ReLU(inplace=True))
            elif relu == 'leaky':
                self.main.append(nn.LeakyReLU(negative_slope=nslope, inplace=True)
        else:

            self.main = nn.ModuleList(
                            norm,

                            nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=bias)
                    )

            if relu == 'relu':
                self.main.insert(0, nn.ReLU(inplace=True))
            elif relu == 'leaky':
                self.main.insert(0, nn.LeakyReLU(negative_slope=nslope, inplace=True)


    def forward(self, x):
        '''
        Method that defines the forward pass through the BNConv network.

        Arguments:
        - x : The input to the network

        Returns:
        - The output of the network BNConv
        '''

        for layer in self.main:
            x = layer(x)

        return x
