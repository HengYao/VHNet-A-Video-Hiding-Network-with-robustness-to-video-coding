import glob
import config as c
import skvideo.io
import numpy as np
import torchvision.transforms as T
import torch
import cv2
import os
import matplotlib.image as mpimg
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset, DataLoader



class Test_data():
    def __init__(self):
        # self.file=sorted(glob.glob(c.TEST_PATH + "/*"))
        self.test_num = c.test_num
        self.cover_list = sorted(glob.glob(c.TEST_PATH_cover + "/*"))[0:self.test_num]
        self.secret_list = sorted(glob.glob(c.TEST_PATH_secret + "/*"))[0:self.test_num]

    def get_data(self, index):
        cover_name=self.cover_list[index][-9:-4]
        secret_name = self.secret_list[index][-9:-4]
        cover = skvideo.io.vread(self.cover_list[index])/255.0
        secret = skvideo.io.vread(self.secret_list[ index])/255.0
        total_frames = np.min((cover.shape[0], secret.shape[0])) // c.clip_len * c.clip_len
        cover=torch.from_numpy(np.transpose(cover[0:total_frames], [3, 0, 1, 2]))
        secret=torch.from_numpy(np.transpose(secret[0:total_frames], [3, 0, 1, 2]))
        cover=cover.unsqueeze(0)
        secret=secret.unsqueeze(0)

        return cover_name,secret_name,total_frames//c.clip_len,cover.float(),secret.float()

