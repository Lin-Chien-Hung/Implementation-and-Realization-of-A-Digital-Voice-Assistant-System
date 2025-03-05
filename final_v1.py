# coding=UTF-8
from __future__ import print_function
import os
import shutil
import argparse
import pandas as pd

import glob
import torch.nn.functional as F
from feature_extraction import extract_MFB
from generator.SR_Dataset import *

from generator.DB_wav_reader import read_feats_structure

# =================================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU device number.")

# =================================================================================================
# Test setting
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

speaker = ["0_David", "Jeffrey", "Elsa", "Cindy", "chien_hung"]
switch = 512
num_enroll_utterance = 6


# =================================================================================================
def remove_existing_files(directory):

    files_to_remove = glob.glob(os.path.join(directory, "*"))

    for file in files_to_remove:

        os.remove(file)


# =================================================================================================
def get_DB(feat_dir):

    DB = pd.DataFrame()
    # 讀取存放 聲學特徵 的資料夾
    for idx, feat_dir in enumerate(feat_dir):

        tmp_DB, len_DB, len_speaker = read_feats_structure(feat_dir, idx)
        # 記錄所有語者pkl檔
        DB = pd.concat([DB, tmp_DB], ignore_index=True)
    return DB


##################################################################################################
def look(model, test_DB, enroll_DB):

    directories_name = []
    score_av_list = []
    # 把那個測試語者底下的所有pkl檔都拉出來，因為只有一個，所以 label = 0
    pair_list = list(test_DB.loc[test_DB["labels"] == 0]["filename"])
    test_filename = pair_list[0]
    # 把 測試語者 的 語者特徵 拉出來
    test_embedding, test_label = read_MFB(test_filename)
    print()
    # 讀取 註冊語者特徵(0 ~ 3)
    for path in glob.glob(os.path.join("./data/enroll/", "*")):
        directories_name.append(os.path.splitext(os.path.basename(path))[0])
        i = 0
        en_ebd = torch.empty(num_enroll_utterance, switch).cuda()
        # 0 ~ 3 各個底下的 *.pkl 檔
        for path_1 in glob.glob(os.path.join(path, "*.pkl")):
            with torch.no_grad():
                speaker_feat, speaker = read_MFB(path_1)
                en_ebd[i] = speaker_feat
                # 刪除上一個的語者特徵
                i += 1
        # 作為度調整
        en_ebd = torch.mean(en_ebd, dim=0).unsqueeze(0)
        # 使用cos相似度比較 註冊語者(語者特徵) 和 測試語者(語者特徵) 的相似度
        score = F.cosine_similarity(en_ebd, test_embedding)
        # 使用 cpu 將數值由 tensor 格式轉換為 numpy 格式
        score = score.data.cpu().numpy()[0]
        # 將分數記錄起來
        score_av_list.append(score)

    os.remove(test_filename)
    # 回傳所有分數
    return score_av_list, directories_name


# =================================================================================================
def pre_enroll_DB():

    enroll_feat_dir = [c.ENROLL_FEAT_DIR]

    return get_DB(enroll_feat_dir)


# =================================================================================================
def defence(score_list, speaker_name_1):

    if os.path.exists("./data/enroll/.ipynb_checkpoints"):
        shutil.rmtree("./data/enroll/.ipynb_checkpoints")

    x_ = np.sort(score_list)[::-1]
    print("x_ = ", x_)
    print("score_list = ", score_list)
    print("np.argmax(score_list) = ", np.argmax(score_list))
    print("speaker_name_1 = ", speaker_name_1)

    if np.max(score_list) > 0.3:
        if np.max(score_list) < 0.63:
            print("語者辨識 :" + "\033[91m" + " Imposter" + "\033[0m")
            return "Imposter"
        else:
            print(
                "語者辨識 :"
                + "\033[92m"
                + " Welcome , "
                + str(speaker_name_1[np.argmax(score_list)])
                + "\033[0m"
            )
            return "Welcome " + str(speaker_name_1[np.argmax(score_list)])
    else:
        print("Spoofing attack")
        return "Spoofing_attack"


# =================================================================================================
def main(model, enroll_DB, test_DB):
    # 模型、測試語者pkl紀錄檔、註冊語者pkl紀錄檔
    center_score, speaker_name_1 = look(model, test_DB, enroll_DB)
    # 輸出分數高的語者
    speaker_name1 = defence(center_score, speaker_name_1)
    return speaker_name1


# =================================================================================================
def speaker_detect(model, audio_data):

    # 註冊語者pkl紀錄檔
    db_enroll = pre_enroll_DB()
    # 刪除上一次錄音的紀錄檔
    remove_existing_files("./data/record/")
    # dtype_info = np.array(audio_data).dtype
    # print('audio_data',dtype_info)
    # 將聲音資訊由 float64 轉換至 float32 的格式
    received_audio_data = (
        audio_data / max(np.abs(audio_data).max(), 1) * 32767
    ).astype(np.int16)
    # dtype_info = np.array(received_audio_data).dtype
    # print('received_audio_data', type(dtype_info))
    # 將輸入的聲音特徵做語者特徵的抽取的動作並儲存起來
    extract_MFB(received_audio_data, model)
    # 獲取測試語者資料夾路徑
    test_feat_dir = [c.TEST_FEAT_DIR]  # save_path + '/test'
    # 測試語者pkl紀錄檔
    test_DB = get_DB(test_feat_dir)
    # 將 註冊語者的特徵檔 及 測試語者的特徵檔 代入模型中進行相似度的比較
    speaker_name1 = main(model, db_enroll, test_DB)
    # 回傳信標回去
    return speaker_name1
