# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import cv2

BASE_DIR = Path('/home/yohei.nomoto/bird2021/solution4birdcall2021')
DATA_DIR = BASE_DIR / 'data'
RAWDATA_DIR = DATA_DIR / 'raw'
NPYDATA_DIR = DATA_DIR / 'processed'
# TRAIN_xst_DATA_DIR下には, BIRDNAME / each_file.ogg (or ogg.npy)
TRAIN_1st_DATA_DIR = NPYDATA_DIR / 'audio_images'
TRAIN_2nd_DATA_DIR = RAWDATA_DIR / 'train_short_audio'

BIRD_NAME = pd.read_csv('birdname.csv').columns.values
BIRD_CODE = {bird_name: i for i, bird_name in enumerate(BIRD_NAME)}

class SpectrogramDataset2ndStage(torch.utils.data.Dataset):
    
    def __init__(
        self,
        file_list: [[str, str]],
        waveform_transforms=None):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.waveform_transforms = waveform_transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        ebird_name, ogg_file_name = self.file_list[idx]
        ogg_path = TRAIN_2nd_DATA_DIR / ebird_name / ogg_file_name
        ebird_code = BIRD_CODE[ebird_name]
        y, sr = sf.read(ogg_path)
        
        # randomly crop 5sec chunk and make prediction
        PERIOD = 5

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * PERIOD
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=np.float64)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float64)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start+effective_length].astype(np.float32)
            else:
                y = y.astype(np.float64)

        y = np.nan_to_num(y)

        labels = np.zeros(len(BIRD_CODE), dtype=float)
        labels[ebird_code] = 1.

        return y, labels


class SpectrogramDataset1stStage(SpectrogramDataset2ndStage):

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        return np.expand_dims(image, 0)

    def __getitem__(self, idx: int):
        ebird_name, ogg_file_name = self.file_list[idx]
        ogg_file_name += '.npy'
        
        # utilizing data from https://www.kaggle.com/kneroma/kkiller-birdclef-2021
        ogg_path = TRAIN_1st_DATA_DIR / ebird_name / ogg_file_name
        ebird_code = BIRD_CODE[ebird_name]
        mel_images = np.load(ogg_path)
        mel_image = mel_images[np.random.choice(len(mel_images))]
        mel_image = self.normalize(mel_image)

        labels = np.zeros(len(BIRD_CODE), dtype=float)
        labels[ebird_code] = 1.

        # mel_image: [1, freq_dim, time_dim]
        return mel_image, labels