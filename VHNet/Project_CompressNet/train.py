import torch
import torch.nn
import torch.optim
import math
import numpy as np
import compressNet
import config as c
from torch.utils.tensorboard.writer import SummaryWriter
import datasets
import warnings
import torch.nn as nn
import torch.nn.init as init
from compressNet import CompressNet


warnings.filterwarnings("ignore")

def comp_loss(output, GT):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    # loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)

    loss = loss_fn(output, GT)
    return loss.to(device)

# torch.nn.L1Loss
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


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)

def initialize_weights(net_l, scale=0.01):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()



#####################
# Model initialize: #
#####################
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # net = compressNet.CompressNet()
    # net = RevNet_3D()
    # net=ResidualDenseBlock(3,3)
    # net = Unet(24,24)
    net=CompressNet()

    net=net.to(device)
    # initialize_weights(net)

    # net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    para = get_parameter_number(net)
    print(para)
    params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
    # initialize_weights(net)
    optim = torch.optim.Adam(params_trainable, lr=c.lr)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    if c.train_next:
        load(c.MODEL_PATH + c.suffix)

    try:
        writer = SummaryWriter(comment='CompressNet')

        for i_epoch in range(c.trained_epoch,c.epochs):
            # i_epoch = i_epoch + c.trained_epoch + 1
            loss_history = []

            #################
            #     train:    #
            #################
            count=0
            for i_batch, (video_ori,video_com) in enumerate(datasets.trainloader):
                # print(i_batch)
                video_ori=video_ori.to(device)
                video_com=video_com.to(device)

                #################
                #    forward:   #
                #################
                output = net(video_ori)

                #################
                #     loss:     #
                #################
                loss = comp_loss(output.cuda(), video_com.cuda())

                loss.backward()
                optim.step()
                optim.zero_grad()

                loss_history.append([loss.item(), 0.])

                # if count%500==0 and count!=0:
                #     print('                       ',np.mean(np.array(loss_history), axis=0))
                #     loss_history = []
                # count=count+1

            epoch_losses = np.mean(np.array(loss_history), axis=0)
            # epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])
            epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])
            print('epoch:',i_epoch,'    loss:',epoch_losses[0])

            #################
            #     val:    #
            #################
            if i_epoch % c.val_freq == 0 & i_epoch != 0:
                with torch.no_grad():
                    psnr_com = []
                    psnr_ori = []
                    net.eval()
                    for (x_ori,x_com) in datasets.valloader:
                        x_ori = x_ori.to(device)
                        x_com = x_com.to(device)

                        output = net(x_ori)

                        x_ori = x_ori.cpu().numpy().squeeze() * 225
                        np.clip(x_ori, 0, 255)
                        output=output.cpu().numpy().squeeze()*225
                        np.clip(output, 0, 255)
                        x_com = x_com.cpu().numpy().squeeze() * 225
                        np.clip(x_com, 0, 255)

                        psnr_temp = computePSNR(output, x_com)
                        psnr_com.append(psnr_temp)

                        psnr_temp_o = computePSNR(output, x_ori)
                        psnr_ori.append(psnr_temp_o)

                    print(np.mean(psnr_com))
                    writer.add_scalars("PSNR_com", {"average psnr": np.mean(psnr_com)}, i_epoch)
                    writer.add_scalars("PSNR_ori", {"average psnr": np.mean(psnr_ori)}, i_epoch)
                    # print(np.mean(psnr_com),'      ',np.mean(psnr_ori))

            writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

            if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
                torch.save({'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % (i_epoch+1) + '.pt')
            weight_scheduler.step()

        torch.save({'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
        writer.close()

    except:
        if c.checkpoint_on_error:
            torch.save({'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT' + '.pt')
        raise


    finally:
        pass