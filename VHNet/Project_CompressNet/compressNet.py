import torch
import torch.nn as nn
import torch.nn.functional as F


class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()
        self.conv0 = nn.Conv3d(3, 64, [1, 1, 1], padding=[0, 0, 0])

        self.conv1 = nn.Conv3d(64, 64, [3, 3, 3], padding=[1, 1, 1])
        self.conv2 = nn.Conv3d(64, 64, [3, 3, 3], padding=[1, 1, 1])
        self.conv3 = nn.Conv3d(64, 64, [3, 3, 3], padding=[1, 1, 1])

        self.conv4 = nn.Conv3d(64, 128, [3, 3, 3], padding=[1, 1, 1])
        self.conv5 = nn.Conv3d(128, 128, [3, 3, 3], padding=[1, 1, 1])
        self.conv6 = nn.Conv3d(128, 128, [3, 3, 3], padding=[1, 1, 1])

        self.conv7 = nn.Conv3d(128, 64, [3, 3, 3], padding=[1, 1, 1])
        self.conv8 = nn.Conv3d(64, 64, [3, 3, 3], padding=[1, 1, 1])
        self.conv9 = nn.Conv3d(64, 64, [3, 3, 3], padding=[1, 1, 1])

        self.conv10 = nn.Conv3d(64, 3, [1, 1, 1], padding=[0, 0, 0])

    def forward(self, x):
        conv0 = self.conv0(x)

        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3_down = F.max_pool3d(conv3, [2, 2, 2], [2, 2, 2])

        conv4 = self.conv4(conv3_down)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        conv6_up = F.interpolate(conv6, scale_factor=(2, 2, 2))

        conv7 = self.conv7(conv6_up)
        conv7_res = conv3 + conv7
        conv8 = self.conv8(conv7_res)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)
        y = x + conv10

        return y
        # return F.relu(y)

