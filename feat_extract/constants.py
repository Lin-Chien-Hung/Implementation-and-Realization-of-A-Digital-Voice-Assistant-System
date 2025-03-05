from configure import save_path

#save_path = '../data'  
# VoxCeleb2 train
TRAIN_AUDIO_VOX2 = save_path+'/voxceleb2/wav'
TRAIN_FEAT_VOX2 = save_path+'/voxceleb2/feat'

# VoxCeleb1 train
TRAIN_AUDIO_VOX1 = save_path+'/voxceleb1/wav'
ENROLL_FEAT_VOX1 = save_path+'/enroll'

# VoxCeleb1 test
TEST_AUDIO_VOX1 = save_path+'/record'
TEST_FEAT_VOX1= save_path+'/test'

USE_LOGSCALE = True
USE_NORM=  True
USE_SCALE = False

SAMPLE_RATE = 16000
FILTER_BANK = 40
