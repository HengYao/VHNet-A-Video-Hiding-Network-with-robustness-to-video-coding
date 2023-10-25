
clamp = 2.0
log10_lr = -5
lr = (10**log10_lr)
init_scale = 0.01
clip_len=8
clip_height=128
clip_width=128
lamda_reconstruction = 1
lamda_guide = 0.8

# Train:
# 1.epoch
batch_size = 6
weight_step = 15 # 20
gamma = 0.5
batchsize_val = 6
val_freq = 20
epochs = 120
SAVE_freq = 5

# 2.iteration
iterations = 140000
weight_step_iterations = 20000
val_freq_iterations = 20000
SAVE_freq_iterations = 20000



TRAIN_PATH = r'D:\sxf\dataset\UCF101\train'
VAL_PATH = r'D:\SXF\dataset\UCF101\validation'
jpeg_gauss_PATH=r'model/jpeggauss26410.pt'
compressnet_PATH=r'model/compressnet26410.pt'
useRel=True

# Saving checkpoints:
MODEL_PATH = 'model/'
checkpoint_on_error = True


# Load:
suffix = 'model_checkpoint_00005.pt'
train_next = False
trained_epoch = 0

# test
TEST_MODEL_PATH=r'model/'
suffix_test='model_checkpoint_00080.pt'
TEST_PATH_cover = r'D:\sxf\dataset\UCF101\test\cover'#####
TEST_PATH_secret = r'D:\sxf\dataset\UCF101\test\secret'#####
test_num=150

