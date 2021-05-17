# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import librosa
import cv2

BASE_DIR = Path('/Users/yohei/Documents/bird2021/solution4birdcall2021')
SMALLDATA_DIR = BASE_DIR / 'data' / 'small'
TRAINDATA_DIR = SMALLDATA_DIR / 'train_short_audio'
LOG_DIR = BASE_DIR / 'reports' / 'logs'
OUTPUT_DIR = BASE_DIR / 'models' / 'output'

BIRD_CODE = {'acafly': 0, 'acowoo': 1}

class SpectrogramDataset(torch.utils.data.Dataset):
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
        ogg_path = TRAINDATA_DIR / ebird_name / ogg_file_name
        ebird_code = BIRD_CODE[ebird_name]
        y, sr = sf.read(ogg_path)
        
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
        labels[BIRD_CODE[ebird_name]] = 1.

        return y, labels
