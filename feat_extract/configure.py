# Feature path
save_path = '../data'        # recommend SSD
TRAIN_FEAT_DIR_2 = save_path + '/voxceleb2/feat/train_logfbank_nfilt40' # train_Vox2
TRAIN_FEAT_DIR_1 = save_path + '/voxceleb1/feat/train_logfbank_nfilt40' # train_Vox1
TEST_FEAT_DIR = save_path + '/test'   # test_Vox1
ENROLL_FEAT_DIR = save_path + '/enroll'   # enroll_Vox1

# Training context window size
NUM_WIN_SIZE = 200 # 200ms == 2 seconds
SHORT_SIZE = 100   # 100ms == 1 seconds

# Settings for feature extraction
USE_NORM = True
USE_SCALE = False

FILTER_BANK = 40