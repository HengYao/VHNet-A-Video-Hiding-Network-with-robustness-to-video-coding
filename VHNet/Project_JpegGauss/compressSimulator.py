from noise import *
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
# from compressNet import CompressNet
import config as c



#########################         测  试        #########################
# def get_parameter_number(net):
#     total_num = sum(p.numel() for p in net.parameters())
#     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
# class Weight(nn.Module):
#     def __init__(self):
#         super(Weight, self).__init__()
#         self.w_gauss = nn.Parameter(torch.FloatTensor([0.5]))
#         self.w_jpeg = nn.Parameter(torch.FloatTensor([0.5]))
#         # w_gauss.requires_grad = True
#         # w_jpeg.requires_grad = True
#
#     def forward(self,x):
#         return 0
#########################         测  试        #########################

def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()

'''
3x(高斯3d+JPEG+高斯2d)  w/o UNET
'''
class CompressSimulator(nn.Module):
    def __init__(self, height=c.clip_height, width=c.clip_width):
        super(CompressSimulator, self).__init__()

        # self.w_gauss3d_l = nn.Parameter(torch.FloatTensor([0.166]))
        self.w_gauss3d = nn.Parameter(torch.FloatTensor([0.333]))
        # self.w_jpeg_l = nn.Parameter(torch.FloatTensor([0.166]))
        self.w_jpeg = nn.Parameter(torch.FloatTensor([0.333]))
        self.w_gauss2d = nn.Parameter(torch.FloatTensor([0.333]))
        # self.w_gauss2d_h = nn.Parameter(torch.FloatTensor([0.166]))

        # self.compressNet = CompressNet()
        # init_model(self.compressNet)

        # self.gaussianBlur3D_h1 = GaussianBlur3D(kernel_size=[3, 5, 5], sigma=1.2, channels=3, clip_len=c.clip_len,
        #                                      height=height, width=width)
        # self.gaussianBlur3D_h2 = GaussianBlur3D(kernel_size=[3, 5, 5], sigma=1.0, channels=3, clip_len=c.clip_len,
        #                                      height=height, width=width)
        self.gaussianBlur3D = GaussianBlur3D(kernel_size=[3, 5, 5], sigma=0.8, channels=3, clip_len=c.clip_len,
                                             height=height, width=width)




        self.diffJpeg_1 = DiffJPEG(differentiable=True, quality=90)
        self.diffJpeg_2 = DiffJPEG(differentiable=True, quality=80)
        self.diffJpeg_3 = DiffJPEG(differentiable=True, quality=70)

        # self.gaussianBlur2D_h1 = GaussianBlur2D(kernel_size=5, sigma=1.2, channels=3 * c.clip_len, height=height, width=width)
        self.gaussianBlur2D = GaussianBlur2D(kernel_size=5, sigma=0.8, channels=3 * c.clip_len, height=height, width=width)
        # self.gaussianBlur2D_h3 = GaussianBlur2D(kernel_size=5, sigma=0.8, channels=3 * c.clip_len, height=height, width=width)


        self.gaussianBlur3D.requires_grad_(False)
        # self.gaussianBlur3D_l2.requires_grad_(False)
        # self.gaussianBlur3D_l3.requires_grad_(False)
        # self.gaussianBlur3D_h1.requires_grad_(False)
        # self.gaussianBlur3D_h2.requires_grad_(False)
        # self.gaussianBlur3D_h3.requires_grad_(False)
        self.diffJpeg_1.requires_grad_(False)
        self.diffJpeg_2.requires_grad_(False)
        self.diffJpeg_3.requires_grad_(False)
        # self.diffJpeg_h1.requires_grad_(False)
        # self.diffJpeg_h2.requires_grad_(False)
        # self.diffJpeg_h3.requires_grad_(False)

        self.gaussianBlur2D.requires_grad_(False)
        # self.gaussianBlur2D_h2.requires_grad_(False)
        # self.gaussianBlur2D_h3.requires_grad_(False)
        # self.gaussianBlur2D_l1.requires_grad_(False)
        # self.gaussianBlur2D_l2.requires_grad_(False)
        # self.gaussianBlur2D_l3.requires_grad_(False)

        self.diffJpeg = [self.diffJpeg_1, self.diffJpeg_2,self.diffJpeg_3]


    def forward(self, x):
        r=torch.randint(0,3,[1])
        x_jpeg = self.diffJpeg[r[0]](x)
        # x_jpeg_h = self.diffJpeg_h[r[1]](x)
        x_gauss3d = self.gaussianBlur3D(x)
        # x_gauss3d_h = self.gaussianBlur3D_h[r[3]](x)
        x_gauss2d = self.gaussianBlur2D(x)
        # x_gauss2d_h = self.gaussianBlur2D_h[r[5]](x)

        y = self.w_jpeg*x_jpeg+self.w_gauss3d*x_gauss3d+self.w_gauss2d*x_gauss2d
        # y = self.compressNet(y)

        return y

