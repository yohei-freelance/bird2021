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

def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_list: [[str, str]], img_size=224,
        waveform_transforms=None, spectrogram_transforms=None, melspectrogram_parameters={}
    ):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters

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

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float64)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

#       labels = np.zeros(len(BIRD_CODE), dtype="i")
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        labels[BIRD_CODE[ebird_name]] = 1.

        return {"image": image, "targets": labels}
