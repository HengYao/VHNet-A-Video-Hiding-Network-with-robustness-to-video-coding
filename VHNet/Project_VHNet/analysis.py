import numpy as np
import skvideo.io
import matplotlib.pyplot as plt
import math
import cv2
import glob


############### MSE RMSE PSNR SSIM  隐写分析*2 ###############
def calculate_apd(img1, img2): #MAE

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)


def calculate_rmse(img1, img2):
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    rmse = np.sqrt(mse)

    return np.mean(rmse)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    GT_C=r'D:\sxf\dataset\UCF101\test/cover'
    GT_S=r'D:\sxf\dataset\UCF101\test/secret'

    path='results101100'
    cover_path=path+'/cover_img'
    secret_path=path+'/secret_img'
    cover_list = sorted(glob.glob(cover_path + "/*"))
    secret_list = sorted(glob.glob(secret_path + "/*"))

    GT_C_list=sorted(glob.glob(GT_C + "/*"))
    GT_S_list=sorted(glob.glob(GT_S + "/*"))


    print(len(cover_list),'对视频')

    apd_C=[]
    rmse_C=[]
    psnr_C=[]
    ssim_C=[]

    apd_S=[]
    rmse_S=[]
    psnr_S=[]
    ssim_S=[]

    for i in range(len(cover_list)):
        cover=[]
        secret=[]
        GT_cover=0
        GT_secret=0
        cover_imgs=glob.glob(cover_list[i] + "/*")
        secret_imgs = glob.glob(secret_list[i] + "/*")

        for j in range(len(cover_imgs)):
            cover.append(cv2.imread(cover_imgs[j]))
            secret.append(cv2.imread(secret_imgs[j]))

            # GT_cover.append(cv2.imread(GT_C_list[j]))
            # GT_secret.append(cv2.imread(GT_S_list[j]))

        cover=np.array(cover).astype(np.float64)
        secret=np.array(secret).astype(np.float64)
        GT_cover=skvideo.io.vread(GT_C_list[i])[0:cover.shape[0],...].astype(np.float64)[...,::-1]
        GT_secret = skvideo.io.vread(GT_S_list[i])[0:cover.shape[0],...].astype(np.float64)[...,::-1]

        psnr_C.append(calculate_psnr(GT_cover,cover))
        psnr_S.append(calculate_psnr(GT_secret,secret))

        # ssim_C.append(np.mean([calculate_ssim(GT_cover[k],cover[k]) for k in range(cover.shape[0])]))
        # ssim_S.append(np.mean([calculate_ssim(GT_secret[k],secret[k]) for k in range(secret.shape[0])]))

    # apd_C = apd_C/len(cover_list)
    # rmse_C = rmse_C/len(cover_list)
    psnr_C2 = np.mean(psnr_C)
    # ssim_C = np.mean(ssim_C)
    #
    # apd_S = apd_S/len(cover_list)
    # rmse_S = rmse_S/len(cover_list)
    psnr_S2 = np.mean(psnr_S)
    # ssim_S = np.mean(ssim_S)

print('----------  PSNR  ---------')
print('Cover:',psnr_C2,'\t\t\tSecret:',psnr_S2)

# print('----------  SSIM  ---------')
# print('Cover:',ssim_C,'\t\t\tSecret:',ssim_S)