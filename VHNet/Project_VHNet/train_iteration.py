#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
from torch.utils.tensorboard.writer import SummaryWriter
import datasets
import warnings
from CS.compressSim import CompressSim
# from RelModule import RelModule

warnings.filterwarnings("ignore")


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)  # (reduction='mean')
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)  # (reduction='mean')
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)  # (reduction='mean')
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


# def load(network, name):
#     state_dicts = torch.load(name)
#     network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
#     network.load_state_dict(network_state_dict)
#     # try:
#     #     optim.load_state_dict(state_dicts['opt'])
#     # except:
#     #     print('Cannot load optimizer for some reason or other')

def load(network, pathname, netname):
    state_dicts = torch.load(pathname)
    network_state_dict = {k: v for k, v in state_dicts[netname].items() if 'tmp_var' not in k}
    network.load_state_dict(network_state_dict)
    # try:
    #     optim.load_state_dict(state_dicts['opt'])
    # except:
    #     print('Cannot load optimizer for some reason or other')


#####################
# Model initialize: #
#####################
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Model()
    net = net.to(device)
    init_model(net)

    # relModule=RelModule(c.clip_len,3,c.useRel)
    # relModule=relModule.to(device)
    # init_model(relModule)

    cs = CompressSim()
    # load(cs,c.MODEL_PATH+c.compressSimulator_pth,'net')
    cs.requires_grad_(False)
    cs = cs.to(device)

    para = get_parameter_number(net)
    print(para)

    # params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

    params_trainable_net = (list(filter(lambda p: p.requires_grad, net.parameters())))
    # params_trainable_rel = (list(filter(lambda p: p.requires_grad, relModule.parameters())))

    optim = torch.optim.Adam(params_trainable_net, lr=c.lr)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step_iterations, gamma=c.gamma)


    if c.train_next:
        load(net, c.MODEL_PATH + c.suffix,'net')
        # load(relModule, c.MODEL_PATH + c.suffix, 'net_rel')

    try:
        writer = SummaryWriter(comment='INN')
        iteration = 0
        for i_epoch in range(c.epochs):
            # i_epoch = i_epoch + c.trained_epoch + 1 #106-182
            loss_history = []

            #################
            #     train:    #
            #################

            for i_batch, data in enumerate(datasets.trainloader):
                # print(i_batch)
                data = data.to(device)
                cover_input = data[data.shape[0] // 2:]
                secret_input = data[:data.shape[0] // 2]

                # secret_input_rel = relModule(secret_input)

                input_img = torch.cat((cover_input, secret_input), 2)  # (1,12,16,64,64)

                #################
                #    forward:   #
                #################
                output = net(input_img)  # b,c,n,h,w
                steg_img = output.narrow(2, 0, c.clip_len)
                output_z = output.narrow(2, c.clip_len, output.shape[2] - c.clip_len)  # (1,12,8,64,64)

                #################
                #   compress:   #
                #################
                # if c.useCompress:
                #     steg_img_compress = net_compress(steg_img)  # (1,3,8,128,128)
                # # 不压缩2
                # else:
                #     steg_img_compress = steg_img
                steg_img_compress = cs(steg_img)


                #################
                #   backward:   #
                #################

                output_z_guass = gauss_noise(output_z.shape)

                output_rev = torch.cat((steg_img_compress, output_z_guass), 2)
                output_image = net(output_rev, rev=True)

                secret_rev = output_image.narrow(2, c.clip_len, output.shape[2] - c.clip_len)

                g_loss = guide_loss(steg_img.cuda(), cover_input.cuda())
                r_loss = reconstruction_loss(secret_rev, secret_input)

                total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss
                total_loss.backward()
                optim.step()
                optim.zero_grad()
                weight_scheduler.step()
                loss_history.append([total_loss.item(), 0.])
                if (iteration % c.SAVE_freq_iterations == 0) & (iteration != 0):
                    epoch_losses = np.mean(np.array(loss_history), axis=0)
                    epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

                    print('iteration:', iteration // 1000, 'K     loss:', epoch_losses[0])
                    loss_history = []
                    writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, iteration)
                    torch.save({'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.6i' % iteration + '.pt')

            #################
            #     val:    #
            #################
            if (iteration % c.val_freq_iterations == 0) & (iteration != 0):
                with torch.no_grad():
                    psnr_s = []
                    psnr_c = []
                    net.eval()
                    cs.eval()
                    # net_compress.eval()
                    for x in datasets.valloader:
                        x = x.to(device)
                        cover_input = x[x.shape[0] // 2:, :, :, :]
                        secret_input = x[:x.shape[0] // 2, :, :, :]

                        # secret_input_rel = relModule(secret_input)

                        input_img = torch.cat((cover_input, secret_input), 2)

                        #################
                        #    forward:   #
                        #################
                        output = net(input_img)  # (1,12,16,64,64)
                        steg_img = output.narrow(2, 0, c.clip_len)  # (1,12,8,64,64)
                        output_z = output.narrow(2, c.clip_len, output.shape[2] - c.clip_len)  # (1,12,8,64,64)

                        #################
                        #   compress:   #
                        #################

                        steg_img_compress = cs(steg_img)

                        #################
                        #   backward:   #
                        #################
                        output_z_guass = gauss_noise(output_z.shape)

                        output_rev = torch.cat((steg_img_compress, output_z_guass), 2)
                        output_image = net(output_rev, rev=True)

                        secret_rev = output_image.narrow(2, c.clip_len, output.shape[2] - c.clip_len)
                        # secret_rev = relModule(secret_rev, True)


                        secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                        np.clip(secret_rev, 0, 255)
                        secret_input = secret_input.cpu().numpy().squeeze() * 255
                        np.clip(secret_input, 0, 255)
                        cover_input = cover_input.cpu().numpy().squeeze() * 255
                        np.clip(cover_input, 0, 255)
                        steg_img = steg_img.cpu().numpy().squeeze() * 255
                        np.clip(steg_img, 0, 255)
                        psnr_temp = computePSNR(secret_rev, secret_input)
                        psnr_s.append(psnr_temp)
                        psnr_temp_c = computePSNR(cover_input, steg_img)
                        psnr_c.append(psnr_temp_c)

                    print("PSNR  cover:", np.mean(psnr_c),'     secret:',np.mean(psnr_s))
                    writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, iteration)
                    writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, iteration)
            if (iteration > c.iterations):
                break
        torch.save({'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
        writer.close()

    except:
        if c.checkpoint_on_error:
            torch.save({'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT' + '.pt')
        raise

    finally:
        pass
