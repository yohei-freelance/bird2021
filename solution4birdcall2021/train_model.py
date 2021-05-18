# main libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import soundfile as sf
import typing as tp
import torchlibrosa
import librosa
from sklearn.model_selection import StratifiedKFold

# sub modules
from pathlib import Path
from glob import glob
import warnings
warnings.simplefilter('ignore')
from catalyst import dl, utils

# original
from src.data import SpectrogramDataset

target_columns = pd.read_csv('birdname.csv').columns.values
num_classes = len(target_columns)

# directories
BASE_DIR = Path('/Users/yohei/Documents/bird2021/solution4birdcall2021')
SMALLDATA_DIR = BASE_DIR / 'data' / 'small'
TRAINDATA_DIR = SMALLDATA_DIR / 'train_short_audio'
LOG_DIR = BASE_DIR / 'reports' / 'logs'
OUTPUT_DIR = BASE_DIR / 'models' / 'output'


if __name__ == '__main__':
    train_meta_path = SMALLDATA_DIR / 'train_metadata.csv'
    train_meta = pd.read_csv(train_meta_path)

    # !ここではテスト用として2種類の鳥のみ扱う!

    train_meta = train_meta.query('primary_label in ["acafly", "acowoo"]')
    BIRD_CODE = {bird_name: i for i, bird_name in enumerate(target_columns)}
    INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # define model
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0')

    model._fc = nn.Sequential(
    nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Linear(512, 2))
    model.to(device)

    # loss function, optimizer, scheduler
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10)

    # validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_meta["fold"] = -1
    for fold_id, (train_index, val_index) in enumerate(skf.split(train_meta, train_meta["primary_label"])):
        train_meta.iloc[val_index, -1] = fold_id

    use_fold = 0

    train_file_list = train_meta.query("fold != @use_fold")[['primary_label', 'filename']].values.tolist()
    valid_file_list = train_meta.query("fold == @use_fold")[['primary_label', 'filename']].values.tolist()
    train_dataset = SpectrogramDataset(file_list=train_file_list)
    valid_dataset = SpectrogramDataset(file_list=valid_file_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=5)
    loaders = {"train": train_loader, "valid": valid_loader}

    runner = dl.SupervisedRunner(device, input_key="image", target_key="targets")
    print('device: {}'.format(device))

    # 一度callbackは省略!
    runner.train(
        model=model,
        criterion=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=30,
        logdir=LOG_DIR,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,)

    # save model
    torch.save(model.state_dict(), 'model_gpu.pth')
    torch.save(model.state_dict().to('cpu'), 'model_cpu.pth')
