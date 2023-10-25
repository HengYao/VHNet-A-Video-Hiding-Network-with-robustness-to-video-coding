import glob
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import config as c
from natsort import natsorted
import skvideo.io
import numpy as np
import torchvision.transforms as T
import torch
import cv2
import matplotlib.image as mpimg
import math

def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


class Video_Dataset(Dataset):
    def __init__(self, mode="train", clip_len=c.clip_len, clip_height=c.clip_height, clip_width=c.clip_width):  # mode: train/val

        self.mode = mode
        self.clip_len = clip_len
        self.clip_height = clip_height
        self.clip_width = clip_width
        if mode == 'train':
            # train
            self.files_ori = sorted(glob.glob(c.TRAIN_PATH_origin + "/*"))
            self.files_com = sorted(glob.glob(c.TRAIN_PATH_recompress + "/*"))
        else:
            # validation
            self.files_ori = sorted(glob.glob(c.VAL_PATH_origin + "/*"))
            self.files_com = sorted(glob.glob(c.VAL_PATH_recompress + "/*"))

    def __getitem__(self, index):
        try:
            video_ori = skvideo.io.vread(self.files_ori[index])
            video_com = skvideo.io.vread(self.files_com[index])

            time_index = np.random.randint(video_ori.shape[0] - self.clip_len + 1)
            height_index = np.random.randint(240 - self.clip_height + 1)
            width_index = np.random.randint(320 - self.clip_width + 1)


            video_ori = video_ori[time_index:time_index + self.clip_len,
                        height_index:height_index + self.clip_height, width_index:width_index + self.clip_width, :]
            video_com = video_com[time_index:time_index + self.clip_len,
                        height_index:height_index + self.clip_height, width_index:width_index + self.clip_width, :]


            video_ori = video_ori / 255.0
            video_com = video_com / 255.0

            video_ori = torch.from_numpy(np.transpose(video_ori, [3, 0, 1, 2]))
            video_com = torch.from_numpy(np.transpose(video_com, [3, 0, 1, 2]))


            return video_ori.float(), video_com.float()  # (C,N,H,W)
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files_ori)


# Training data loader
trainloader = DataLoader(
    Video_Dataset(mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True
)
# Test data loader
valloader = DataLoader(
    Video_Dataset(mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True
)
