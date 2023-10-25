# # # Super parameters
# clamp = 2.0
log10_lr = -4
lr = 10**log10_lr
epochs = 40
# weight_decay = 0
init_scale = 0.01
clip_len=8
clip_height=128
clip_width=128

# channels_in = 3*clip_len
# #
# beta=0.75
# lamda_reconstruction = 5
# lamda_guide = 1
# lamda_low_frequency = 1
# # device_ids = [0]
# #
# # # Train:
batch_size = 6#12
# # cropsize = 224
# betas = (0.5, 0.999)
weight_step = 10
gamma = 1 # 0.5
# #
# # # Val:
# # cropsize_val = 1024
batchsize_val = 1#6
# # shuffle_val = False
val_freq = 50
# #
# #
# # Dataset
TRAIN_PATH_origin = r'D:\sxf\dataset\UCF101\train'
TRAIN_PATH_recompress = r'D:\sxf\dataset\UCF101_crf10\train'
VAL_PATH_origin = r'D:\sxf\dataset\UCF101\validation'
VAL_PATH_recompress = r'D:\sxf\dataset\UCF101_crf10\validation'
# format_train = 'avi'
# format_val = 'avi'
#
# #
# # # Display and logging:
# # loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
# # silent = False
# # live_visualization = False
# # progress_bar = False
# #
# #
# # # Saving checkpoints:
# #
MODEL_PATH = 'model/'
checkpoint_on_error = True
SAVE_freq = 5
# #
# # IMAGE_PATH = 'image/'
# # IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
# # IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
# # IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
# # IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'
# #
# # # Load:
suffix = 'model.pt'
train_next = False
trained_epoch = 0

