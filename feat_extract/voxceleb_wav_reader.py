import os
import sys
import numpy as np
import pandas as pd
from glob import glob

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def find_wavs(directory, pattern='**/*.wav'):
    """Recursively finds all waves matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def read_voxceleb_structure(directory, data_type):
    
    voxceleb = pd.DataFrame()
    
    if data_type == 'wavs':
        voxceleb['filename'] = find_wavs(directory)
    else:
        raise NotImplementedError
    
    voxceleb['filename'] = voxceleb['filename'].apply(lambda x: x.replace('\\', '/'))
    
    return voxceleb
