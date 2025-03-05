import os
import shutil
import numpy as np
import scipy.io as sio

from generator.SR_Dataset import *
from torch.autograd import Variable
from feat_extract import constants as c
from feat_extract.voxceleb_wav_reader import read_voxceleb_structure

import pickle
from python_speech_features import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
speaker_name = 'test01'

#=================================================================================================
def normalize_frames(m,Scale=False):
    # Z-score Scaling（標準分數縮放）： 將數據縮放為其標準分數，即將數據減去平均值並除以標準差。
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis = 0))
#=================================================================================================
def get_d_vector(total_features, model):
    resnest = model
    # 將資訊轉成 tensor 的格式
    TT = ToTensorTestInput()
    inputx = TT(total_features)
    input = Variable(inputx)

    with torch.no_grad():
        input = input.cuda()
        # 把聲學特徵丟進去模型裡面，產語者特徵
        activation = resnest(input)

    return activation
#=================================================================================================
def extract_MFB(received_audio_data, model):
    resnest = model
    # 聲學特徵、計算能量  = filter bank 設計 (音檔資訊 、 取樣平率 、 幾個濾波器 、 寬度多少 、 使用 hamming 濾波器)
    features, energies = fbank(received_audio_data, samplerate=c.SAMPLE_RATE, nfilt=c.FILTER_BANK, winlen=0.025, winfunc=np.hamming)
    # 是否要對聲音資取 log
    if c.USE_LOGSCALE:
        features = 20 * np.log10(np.maximum(features,1e-5))
    # 是否要對聲音資訊取 正規化
    if c.USE_NORM:
        # 是否要對聲音資訊進行縮放
        total_features = normalize_frames(features, Scale=c.USE_SCALE)
    else:
        total_features = features

    # 抽取語者特徵
    activation = get_d_vector(total_features,resnest)

    # 記錄語者名稱
    speaker_label = "guest"
    # 將語者的名稱及聲學特徵放在一起，並於下方將其儲存起來
    feat_and_label = {'feat':activation, 'label':speaker_label}
    # 若資料要儲存的資料夾不存在，則創建
    if not os.path.exists("./data/test/test01"):
        os.makedirs("./data/test/test01")
    # 將聲學特徵儲存起來
    with open("./data/test/test01/guest.pkl", 'wb') as fp:
        pickle.dump(feat_and_label, fp)
        
#=================================================================================================
def extract_MFB_enroll(audio,model,i,register_name):
    
    resnest = model

    # 聲學特徵、計算能量  = filter bank 設計 (音檔資訊 、 取樣率 、 幾個濾波器 、 寬度多少 、 使用 hamming 濾波器)
    features, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=c.FILTER_BANK, winlen=0.025, winfunc=np.hamming)

    # 是否要對聲音資取 log
    if c.USE_LOGSCALE:
        features = 20 * np.log10(np.maximum(features, 1e-5))
    # 是否要對聲音資訊取 正規化
    if c.USE_NORM:
        # 是否要對聲音資訊進行縮放
        total_features = normalize_frames(features, Scale=c.USE_SCALE)
    else:
        total_features = features
#===========================================================================================    
    # 獲得語者名稱
    spk_name = register_name
    
    make_file_name = "./data/enroll/" + spk_name
    if not os.path.exists(make_file_name):
        os.makedirs(make_file_name)
    
    output_foldername = make_file_name
    output_filename = make_file_name + "/" + str(spk_name) + str(i) + ".pkl"
#===========================================================================================
    # 抽取語者特徵
    activation = get_d_vector(total_features, resnest)

    speaker_label = spk_name
    feat_and_label = {'feat': activation, 'label': speaker_label}

    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)

    with open(output_filename, 'wb') as fp:
        pickle.dump(feat_and_label, fp)
#=================================================================================================
def feat_extraction(model, dataroot_dir):

    DB = read_voxceleb_structure(dataroot_dir, data_type='wavs')

    for i in range(len(DB)):

        extract_MFB_enroll(DB['filename'][i],model)

    print("-" * 20 + " Feature extraction done " + "-" * 20)
