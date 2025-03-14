import os
import numpy as np
from feat_extract import constants as c
from feat_extract.voxceleb_wav_reader import read_voxceleb_structure
# import constants as c
# from voxceleb_wav_reader import read_voxceleb_structure
import pickle # For python3
from python_speech_features import *


import scipy.io as sio
import scipy.io.wavfile
from torch import random

speaker_name = 'test01'
def convert_wav_to_MFB_name(filename, mode):
    """
    Converts the wav dir (in DB folder) to feat dir(in feat folder)
    ex) input         : '.../voxceleb/voxceleb1/dev/wav/id10918/oT62hV9eoHo/00001.wav'
    output_foldername : '.../voxceleb/voxceleb1/dev/feat/train_logfbank_nfilt40/id10918/oT62hV9eoHo'
    output_filename   : '.../voxceleb/voxceleb1/dev/feat/train_logfbank_nfilt40/id10918/oT62hV9eoHo/00001.pkl'
    """
    filename_only = filename.split('/')[-1].replace('.wav','.pkl') # ex) 00001.pkl (pickle format)
    
    # uri_folder = filename.split('/')[-2]                           # ex) oT62hV9eoHo
    # speaker_folder = filename.split('/')[-3]                       # ex) id10918
    if c.USE_LOGSCALE == True:
        feature_type = 'logfbank'
    elif c.USE_LOGSCALE == False:
        feature_type = 'fbank'

    if mode == 'enroll':
        output_foldername = os.path.join(c.ENROLL_FEAT_VOX1, speaker_name)
        
    elif mode == 'test':
        output_foldername = os.path.join(c.TEST_FEAT_VOX1, speaker_name) ###改
    
    output_filename = os.path.join(output_foldername, filename_only)
    return output_foldername, output_filename


def extract_MFB(filename, mode):
    sr, audio = sio.wavfile.read(filename)
    features, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=c.FILTER_BANK, winlen=0.025, winfunc=np.hamming)

    if c.USE_LOGSCALE:
        features = 20 * np.log10(np.maximum(features,1e-5))
        
    if c.USE_DELTA:
        delta_1 = delta(features, N=2)  #N原本是1
        delta_2 = delta(delta_1, N=2) #N原本是1
        
        features = normalize_frames(features, Scale=c.USE_SCALE)
        delta_1 = normalize_frames(delta_1, Scale=c.USE_SCALE)
        delta_2 = normalize_frames(delta_2, Scale=c.USE_SCALE)
        features = np.hstack([features, delta_1, delta_2])

    if c.USE_NORM:
        features = normalize_frames(features, Scale=c.USE_SCALE)
        total_features = features

    else:
        total_features = features

        
    speaker_folder = filename.split('/')[-3]
    output_foldername, output_filename = convert_wav_to_MFB_name(filename, mode=mode)

    speaker_label = speaker_folder # set label as a folder name (recommended). Convert this to speaker index when training
    feat_and_label = {'feat':total_features, 'label':speaker_label}

    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)

    if os.path.isfile(output_filename) == 1:
        print("\"" + '/'.join(output_filename.split('/')[-3:]) + "\"" + " file already extracted!")
    else:
        with open(output_filename, 'wb') as fp:
            pickle.dump(feat_and_label, fp)


def normalize_frames(m,Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))


class mode_error(Exception):
    def __str__(self):
        return "Wrong mode (type 'train' or 'test')"

def feat_extraction(dataroot_dir, mode):
    DB = read_voxceleb_structure(dataroot_dir, data_type='wavs')

    if (mode != 'enroll') and (mode != 'test'):
      raise mode_error
    count = 0
    
    for i in range(len(DB)):
        extract_MFB(DB['filename'][i], mode=mode)
        count = count + 1
        filename = DB['filename'][i]
        print("feature extraction (%s DB). step : %d, file : \"%s\"" %(mode, count, '/'.join(filename.split('/')[-3:])))
        tar_filename = DB['filename'][i].split('/')[-1].replace('.wav','.pkl')

    print("-"*20 + " Feature extraction done " + "-"*20)

if __name__ == '__main__':
    tar_filename = feat_extraction(dataroot_dir=c.TEST_AUDIO_VOX1, mode='enroll')      #enroll, test