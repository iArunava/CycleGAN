import torch
import torch.nn as nn

def bnconv(in_channels, out_channels, kernel_size, stride,
             padding, eps=1e-5, momentum=0.1, conv_first=True, relu='relu', nslope=0.2, norm='instance'):
    
    norm = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
    acti = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(negative_slope=nslope, inplace=True)
    bias = norm == nn.InstanceNorm2d

    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
              padding=padding, bias=bias), norm(out_channels, eps=eps, momentum=momentum), acti]
    
    return layers if relu is not None else layers[:2]

def bnconvt(in_channels, out_channels, kernel_size, stride,
             padding, eps=1e-5, momentum=0.1, conv_first=True, relu='relu', nslope=0.2, norm='instance'):

    norm = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
    acti = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(negative_slope=nslope, inplace=True)
    bias = norm == nn.InstanceNorm2d

    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
              stride=stride, padding=1, output_padding=1, bias=bias), norm(out_channels, eps=eps, momentum=momentum), acti]
    
    return layers if relu is not None else layers[:2]

class ResidualBlock2L(nn.Module):
  
  def __init__(self, ic_conv, oc_conv, expansion=4, stride=2, downsample=False,
                     conv_first=True, relu_after_add=True, norm='instance',
                     pad_type='reflect', eps=1e-5, momentum=0.1):
    '''
    This class defines the Residual Basic Block with 2 Conv Layers

    Arguments;
    - ic_conv : # of input channels
    - oc_conv : # of output channels for the final conv layers
    - downsample : Whether this block is to downsample the input
    - expansion : The expansion for the channels
                  Default: 4
    - stride : if downsample is True then the specified stride is used.
               Default: 2
    - conv_first : Whether to apply conv before bn or otherwise.
                   Default: True

	- relu_after_add : Whether to apply the relu activation after
					   adding both the main and side branch.
				       Default: True
    '''
    
    super(ResidualBlock2L, self).__init__()
    
    assert(downsample == True or downsample == False)
    assert(relu_after_add == True or relu_after_add == False)
    self.downsample = downsample
    self.expansion = expansion
    self.relu_after_add = relu_after_add
    oc_convi = oc_conv // self.expansion
    
    stride = stride if self.downsample else 1
     
    layers = []
    
    p = 0
    if pad_type == 'reflect':
        layers += [nn.ReflectionPad2d(1)]
    elif pad_type == 'replicate':
        layers += [nn.ReplicationPad2d(1)]
    elif pad_type == 'pad':
        p = 1
    else:
        raise NotImplemented('This pad type is not implemented!')
    
    layers += [*bnconv(in_channels=ic_conv,
                            out_channels=oc_convi,
                            kernel_size=3,
                            padding=p,
                            stride=stride,
                            #eps=2e-5,
                            #momentum=0.9,
                            conv_first=True,
                            norm=norm,
                            eps=eps,
                            momentum=momentum)]
    
    p = 0
    if pad_type == 'reflect':
        layers += [nn.ReflectionPad2d(1)]
    elif pad_type == 'replicate':
        layers += [nn.ReplicationPad2d(1)]
    elif pad_type == 'pad':
        p = 1
    else:
        raise NotImplemented('This pad type is not implemented!')
    
    layers += [*bnconv(in_channels=oc_convi,
                        out_channels=oc_conv,
                        kernel_size=3,
                        padding=p,
                        stride=1,
                        #eps=2e-5,
                        #momentum=0.9,
                        conv_first=True,
                        relu=None,
                        norm=norm,
                        eps=eps,
                        momentum=momentum)
               ]
    
    
    self.side = nn.Sequential(*layers)
    
  def forward(self, x): return x + self.side(x)

class CycleGAN(nn.Module):

    def __init__(self, ic_conv=64, norm='instance', pad_type='reflect', 
                 eps=1e-5, momentum=0.1):
        '''
        This class defines the network used for CycleGAN.
        This network is solely taken from the Supplementary material provided
        along with the Perceptual Losses paper by Johnson et al.
        '''
        
        super(CycleGAN, self).__init__()
        
        self.net = nn.Sequential(
                        nn.ReflectionPad2d(3),
                        
                        *bnconv(3, ic_conv, kernel_size=7, padding=0, stride=1,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconv(ic_conv, ic_conv, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconv(ic_conv, ic_conv*2, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconv(ic_conv*2, ic_conv*4, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        ResidualBlock2L(ic_conv*4, ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),
                        
                        ResidualBlock2L(ic_conv*4, oc_conv=ic_conv*4,
                                        conv_first=True,
                                        relu_after_add=False,
                                        norm=norm,
                                        pad_type=pad_type,
                                        eps=eps,
                                        momentum=momentum),

                        *bnconvt(ic_conv*4, ic_conv*2, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconvt(ic_conv*2, ic_conv, kernel_size=3, padding=1, stride=2,
                                conv_first=True, norm=norm, eps=eps, momentum=momentum),

                        *bnconvt(ic_conv, ic_conv, kernel_size=3, padding=1, stride=2,
                                conv_first=True, relu=False, norm=norm, eps=eps, momentum=momentum),
                        
                        nn.ReflectionPad2d(3), nn.Conv2d(ic_conv,3, kernel_size=7, padding=0, stride=1),
                        nn.Tanh()
            )

        self.reset_params()

    def forward(self, x): return self.net(x)
    
    def reset_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(layer.bias.data, 0)

class Discriminator(nn.Module):

    def __init__(self, iconv=64, num_conv=4):
        
        super(Discriminator, self).__init__()
        
        layers = [nn.Conv2d(3, iconv, kernel_size=4, stride=2, padding=1), 
                  nn.LeakyReLU(negative_slope=0.2, inplace=True)]

        for _ in range(num_conv-2):
            layers += bnconv(iconv, iconv*2, kernel_size=4, stride=2, padding=1,
                                  relu='leaky', nslope=0.2)
            iconv *= 2

        layers += [*bnconv(iconv, iconv*2, kernel_size=4, stride=1, padding=1, relu='leaky', nslope=0.2),
                   nn.Conv2d(iconv*2, 1, kernel_size=4, stride=1, padding=1)]
        
        self.net = nn.Sequential(*layers)
        #self.reset_params()
        
    def forward(self, x): return self.net(x)
    
    def reset_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(layer.bias.data, 0)
                
    def set_requires_grad(self, trfalse):
        for layer in self.parameters():
            layer.requires_grad = trfalse
