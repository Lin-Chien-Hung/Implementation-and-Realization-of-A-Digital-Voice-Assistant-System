import os
from glob import glob
import sys

import numpy as np
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def find_feats(directory, pattern='*/*.pkl'):
    return glob(os.path.join(directory, pattern), recursive=True)

def read_feats_structure(directory, test=False):
    # 將路近指定過去
    DB = pd.DataFrame()
    DB['filename'] = find_feats(directory) # filename
    DB['filename'] = DB['filename'].unique().tolist()
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) # normalize windows paths
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-2]) # speaker folder name
    # 將語者進行排序，並計算長度，speaker_list ['0_David', '1_Jeffrey', '2_Elsa', '3_Cindy']
    speaker_list = sorted(set(DB['speaker_id']))  # len(speaker_list) == n_speakers
    if test: spk_to_idx = {spk: i+1211 for i, spk in enumerate(speaker_list)}
    else: spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    # 將語者記錄起來
    DB['labels'] = DB['speaker_id'].apply(lambda x: spk_to_idx[x])  # dataset folder name
    num_speakers = len(DB['speaker_id'].unique())
    # 記錄所有語者pkl檔、所有語者的檔案數、幾位語者
    return DB, len(DB), num_speakers

