import skvideo.io
import torch
import matplotlib.pyplot as plt
import glob
import os
import PIL.Image as img
import cv2
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from JPEG_utils import diff_round, quality_to_factor, Quantization
from compression import compress_jpeg
from decompression import decompress_jpeg


# 1.高斯噪声 (0,0.02)
class GaussianNoise(nn.Module):
    def __init__(self,loc=0,scale=0.02):
        super(GaussianNoise, self).__init__()
        self.loc=loc
        self.scale=scale
    def forward(self,input):
        shape = input.shape
        noise = np.random.normal(loc=self.loc, scale=self.scale, size=shape)
        noise = torch.from_numpy(noise).cuda()
        output = input + noise
        return output


# 2.2D高斯滤波 (0,2),5x5
class GaussianBlur2D(nn.Module):
    def __init__(self, kernel_size=5, sigma=0.5, channels=3 * 8, height=128, width=128):  # 3*c.clip_len
        super(GaussianBlur2D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.height = height
        self.width = width
        self.weight = self.gaussian_kernel()
        self.mask = self.weight_mask()

    def gaussian_kernel(self):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size))
        radius = self.kernel_size // 2
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v
        kernel2d = kernel
        kernel = kernel / np.sum(kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        weight = nn.Parameter(data=kernel, requires_grad=False)

        return weight

    def weight_mask(self):
        ones = torch.ones([1, self.channels, self.height, self.width])
        mask = F.conv2d(ones, self.weight, bias=None, stride=1, padding=self.kernel_size // 2, dilation=1,
                        groups=self.channels)
        return mask

    def forward(self, input):
        ### 转b,cn,h,w
        input = torch.concat(input.unbind(2), 1)
        output = F.conv2d(input, self.weight, bias=None, stride=1, padding=self.kernel_size // 2, dilation=1,
                          groups=self.channels)
        output = output / self.mask.cuda()
        b, cn, h, w = output.shape

        return output.view(b, cn // 3, 3, h, w).transpose(1, 2).cuda()


# 3.3D高斯滤波 (0,2),5x5x3
class GaussianBlur3D(nn.Module):
    def __init__(self, kernel_size=[3, 5, 5], sigma=0.5, channels=3, clip_len=8, height=128, width=128):
        super(GaussianBlur3D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.height = height
        self.width = width
        self.clip_len = clip_len
        self.weight = self.gaussian_kernel()
        self.mask = self.weight_mask()

    def gaussian_kernel(self):
        kernel = np.zeros(shape=self.kernel_size)
        radius1 = self.kernel_size[0] // 2
        radius2 = self.kernel_size[1] // 2
        for z in range(-radius1, radius1 + 1):
            for y in range(-radius2, radius2 + 1):
                for x in range(-radius2, radius2 + 1):
                    v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(
                        -1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2 + z ** 2))
                    kernel[z + radius1, y + radius2, x + radius2] = v
        kernel = kernel / np.sum(kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        weight = nn.Parameter(data=kernel, requires_grad=False)
        return weight

    def weight_mask(self):
        ones = torch.ones([1, self.channels, self.clip_len, self.height, self.width])
        mask = F.conv3d(ones, self.weight, bias=None, stride=1,
                        padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2],
                        dilation=1, groups=self.channels)
        return mask

    def forward(self, input):
        output = F.conv3d(input, self.weight, bias=None, stride=1,
                          padding=[self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2],
                          dilation=1, groups=self.channels)
        output = output / self.mask.cuda()
        return output


# 4.可微jpeg q=75
class DiffJPEG(nn.Module):
    def __init__(self, differentiable=True, quality=75):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image height
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
            # rounding = Quantization()
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        # self.decompress = decompress_jpeg(height, width, rounding=rounding,
        #                                   factor=factor)
        self.decompress = decompress_jpeg(rounding=rounding, factor=factor)

    def forward(self, x):
        '''
        '''

        # B,C,N,H,W -> BN,C,H,W ->B,C,N,H,W
        b,c,n,h,w=x.shape#
        x=torch.concat(x.unbind(2),0)#


        org_height = x.shape[2]
        org_width = x.shape[3]
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr, org_height, org_width)

        recovered=recovered.view([n,b,c,h,w]).transpose(0,1).transpose(1,2)#
        return recovered



