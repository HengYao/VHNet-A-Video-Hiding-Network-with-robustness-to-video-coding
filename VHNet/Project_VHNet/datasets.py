import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import config as c
from natsort import natsorted
import skvideo.io
import numpy as np
import torchvision.transforms as T
import torch
import cv2
import os
import matplotlib.image as mpimg


class Video_Dataset(Dataset):
    def __init__(self, mode="train", clip_len=c.clip_len, clip_height=c.clip_height,
                 clip_width=c.clip_width):  # mode: train/val

        self.mode = mode
        self.clip_len = clip_len
        self.clip_height = clip_height
        self.clip_width = clip_width
        if mode == 'train':
            # train
            self.files = sorted(glob.glob(c.TRAIN_PATH + "/*"))
        else:
            # test
            self.files = sorted(glob.glob(c.VAL_PATH + "/*"))

    def __getitem__(self, index):
        try:

            video = skvideo.io.vread(self.files[index])

            time_index = np.random.randint(video.shape[0] - self.clip_len + 1)
            height_index = np.random.randint(240 - self.clip_height + 1)
            width_index = np.random.randint(320 - self.clip_width + 1)

            video = video[time_index:time_index + self.clip_len, height_index:height_index + self.clip_height,
                        width_index:width_index + self.clip_width, :]

            video=video/255

            video = torch.from_numpy(np.transpose(video, [3, 0, 1, 2]))  # 转(C,N,H,W) 并toTensor
            # video = torch.cat(video.unbind(1), 0)  # (C*N,H,W)

            return video.float()  # (C,N,H,W)
            # return video.float()  # (C*N,H,W) (24,128,128)
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)


# Training data loader
trainloader = DataLoader(
    Video_Dataset(mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
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
#输出(B,C,N,H,W) (2,3,8,128,128)