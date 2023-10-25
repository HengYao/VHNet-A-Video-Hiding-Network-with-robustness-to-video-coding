import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
# from torch.utils.tensorboard.writer import SummaryWriter
# import datasets
# import viz
# import warnings
# from compressNet import CompressNet
# import glob
import skvideo.io
import test_datasets
# import matplotlib.pyplot as plt
import os
# from RelModule import RelModule
import cv2



def load(network, pathname, netname):
    state_dicts = torch.load(pathname)
    network_state_dict = {k: v for k, v in state_dicts[netname].items() if 'tmp_var' not in k}
    network.load_state_dict(network_state_dict)
    # try:
    #     optim.load_state_dict(state_dicts['opt'])
    # except:
    #     print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Model()
    net = net.to(device)
    # relModule = RelModule(c.clip_len, 3, c.useRel)
    # relModule = relModule.to(device)
    load(net, c.TEST_MODEL_PATH + c.suffix_test, 'net')
    # load(relModule, c.TEST_MODEL_PATH + c.suffix_test, 'net_rel')
    net.requires_grad_(False)
    # relModule.requires_grad_(False)
    # files = sorted(glob.glob(c.TEST_PATH + "/*"))

    test_data = test_datasets.Test_data()
    os.makedirs('results/cover_img', exist_ok=True)
    os.makedirs('results/secret_img', exist_ok=True)
    os.makedirs('results/cover_vid', exist_ok=True)
    os.makedirs('results/secret_vid', exist_ok=True)
    path_cover_img = 'results/cover_img'
    path_secret_img = 'results/secret_img'
    path_cover_vid = 'results/cover_vid'
    path_secret_vid = 'results/secret_vid'
    with torch.no_grad():
        for i in range(c.test_num):
            cover_name, secret_name, iter_times, cover_vid, secret_vid = test_data.get_data(i)
            os.makedirs(path_cover_img + '/'+ cover_name,exist_ok=True)
            os.makedirs(path_secret_img + '/'+ secret_name,exist_ok=True)
            cover_path = path_cover_img + '/'+ cover_name
            secret_path = path_secret_img + '/'+ secret_name
            # 正向
            img_list_cover = []
            for j in range(iter_times):
                cover_input = cover_vid[:, :, j * c.clip_len:j * c.clip_len + c.clip_len, :, :].to(device)
                secret_input = secret_vid[:, :, j * c.clip_len:j * c.clip_len + c.clip_len, :, :].to(device)


                # secret_input_rel = relmode(
                #         secret_input)
                # secret_input = relModule(secret_input)
                input_img = torch.cat((cover_input, secret_input), 2)  # (1,12,16,64,64)

                #################
                #    forward:   #
                #################
                output = net(input_img)  # b,c,n,h,w
                steg_img = output.narrow(2, 0, c.clip_len)
                output_z = output.narrow(2, c.clip_len, output.shape[2] - c.clip_len)  # (1,12,8,64,64)

                #################
                #   save:   #
                #################

                steg_img = steg_img.cpu().numpy().squeeze() * 255
                steg_img = steg_img.transpose([1,2,3,0])
                img_list_cover.append(steg_img)
            img_list_cover = np.concatenate(img_list_cover, 0)
            img_list_cover = np.clip(img_list_cover, 0, 255)
            img_list_cover=img_list_cover.astype(int)
            ###save img_list###
            img_list_cover2 = img_list_cover[..., ::-1]
            for k in range(img_list_cover2.shape[0]):
                cv2.imwrite(cover_path+ '/%04d.png' % k,img_list_cover2[k])
            os.system("ffmpeg -i {a}/%04d.png -c:v h264 -crf 10 -pix_fmt yuv420p {b}/{c}.mp4".format(a=cover_path,
                                                                                                           b=path_cover_vid,
                                                                                                           c=cover_name))

            # steg_vid_compress=img_list_cover/255.0
            steg_vid_compress = skvideo.io.vread(path_cover_vid + '/' + cover_name+'.mp4') / 255.0

            steg_vid_compress = torch.from_numpy(np.transpose(steg_vid_compress, [3, 0, 1, 2]))
            steg_vid_compress = steg_vid_compress.unsqueeze(0).float()
            img_list_secret = []
            for l in range(iter_times):
                steg_img_compress = steg_vid_compress[:, :, l * c.clip_len:l * c.clip_len + c.clip_len, :, :].to(device)

                #################
                #   backward:   #
                #################

                output_z_guass = gauss_noise(output_z.shape)  # (1,12,8,64,64)

                output_rev = torch.cat((steg_img_compress, output_z_guass), 2)  # (1,12,16,64,64)
                output_image = net(output_rev, rev=True)  # (1,12,16,64,64)

                secret_rev = output_image.narrow(2, c.clip_len, output.shape[2] - c.clip_len)  # (1,12,8,64,64)

                secret_rev_rel = secret_rev

                secret_rev_rel = secret_rev_rel.cpu().numpy().squeeze() * 255
                secret_rev_rel = secret_rev_rel.transpose([1, 2, 3, 0])

                img_list_secret.append(secret_rev_rel)
            img_list_secret = np.concatenate(img_list_secret, 0)
            img_list_secret = np.clip(img_list_secret, 0, 255)
            img_list_secret=img_list_secret.astype(int)
            ###save img_list###
            img_list_secret = img_list_secret[..., ::-1]
            for m in range(img_list_secret.shape[0]):
                cv2.imwrite(secret_path+ '/%04d.png' % m,img_list_secret[m])
            # os.system(
            #     "ffmpeg -i {a}/%04d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p {b}/{c}.mp4".format(a=secret_path,
            #                                                                                          b=path_secret_vid,
            #                                                                                          c=secret_name))
