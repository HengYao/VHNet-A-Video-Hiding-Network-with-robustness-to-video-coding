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
from compressNet import CompressNet
import config as c
from compressSimulator import CompressSimulator


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()

def load(name,network):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    network.load_state_dict(network_state_dict)

'''

'''
class CompressSim(nn.Module):
    def __init__(self, height=c.clip_height, width=c.clip_width):
        super(CompressSim, self).__init__()
        self.jpeg_gauss_rand=CompressSimulator()
        self.net=CompressNet()
        load(c.jpeg_gauss_PATH,self.jpeg_gauss_rand)
        load(c.compressnet_PATH,self.net)
        self.jpeg_gauss_rand.requires_grad_(False)
        self.net.requires_grad_(False)

        self.cs = [self.jpeg_gauss_rand, self.net]

    def forward(self, x):
        r=torch.randint(0,2,[1])
        y=self.cs[r[0]](x)

        return y
