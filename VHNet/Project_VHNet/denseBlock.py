import torch
import torch.nn as nn
import torch.nn.init as init
# import modules.module_util as mutil

# def initialize_weights(net_l, scale=1):
#     if not isinstance(net_l, list):
#         net_l = [net_l]
#     for net in net_l:
#         for m in net.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 m.weight.data *= scale  # for residual block
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias.data, 0.0)

# Dense connection
class ResidualDenseBlock(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock, self).__init__()
        cn=32
        self.conv1 = nn.Conv2d(input, cn, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + cn, cn, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * cn, cn, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * cn, cn, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * cn, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)


        # self.conv1=nn.Conv3d(input,32,3,1,1)


        # initialization
        # initialize_weights([self.conv5], 0.) #conv5 初始化为0

    def forward(self, x):

        #3D (B,C*N,H,W)
        # x = x.unsqueeze(2) #(B,C*N,1,H,W)
        # x=x.split(3,1)
        # x = torch.concat(x,2) #(B,C,N,H,W)

        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
