# main libraries
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import cv2
import soundfile as sf
import typing as tp
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

target_columns = [
    'acafly', 'acowoo', 'aldfly', 'ameavo', 'amecro',
    'amegfi', 'amekes', 'amepip', 'amered', 'amerob',
    'amewig', 'amtspa', 'andsol1', 'annhum', 'astfly',
    'azaspi1', 'babwar', 'baleag', 'balori', 'banana',
    'banswa', 'banwre1', 'barant1', 'barswa', 'batpig1',
    'bawswa1', 'bawwar', 'baywre1', 'bbwduc', 'bcnher',
    'belkin1', 'belvir', 'bewwre', 'bkbmag1', 'bkbplo',
    'bkbwar', 'bkcchi', 'bkhgro', 'bkmtou1', 'bknsti', 'blbgra1',
    'blbthr1', 'blcjay1', 'blctan1', 'blhpar1', 'blkpho',
    'blsspa1', 'blugrb1', 'blujay', 'bncfly', 'bnhcow', 'bobfly1',
    'bongul', 'botgra', 'brbmot1', 'brbsol1', 'brcvir1', 'brebla',
    'brncre', 'brnjay', 'brnthr', 'brratt1', 'brwhaw', 'brwpar1',
    'btbwar', 'btnwar', 'btywar', 'bucmot2', 'buggna', 'bugtan',
    'buhvir', 'bulori', 'burwar1', 'bushti', 'butsal1', 'buwtea',
    'cacgoo1', 'cacwre', 'calqua', 'caltow', 'cangoo', 'canwar',
    'carchi', 'carwre', 'casfin', 'caskin', 'caster1', 'casvir',
    'categr', 'ccbfin', 'cedwax', 'chbant1', 'chbchi', 'chbwre1',
    'chcant2', 'chispa', 'chswar', 'cinfly2', 'clanut', 'clcrob',
    'cliswa', 'cobtan1', 'cocwoo1', 'cogdov', 'colcha1', 'coltro1',
    'comgol', 'comgra', 'comloo', 'commer', 'compau', 'compot1',
    'comrav', 'comyel', 'coohaw', 'cotfly1', 'cowscj1', 'cregua1',
    'creoro1', 'crfpar', 'cubthr', 'daejun', 'dowwoo', 'ducfly', 'dusfly',
    'easblu', 'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eletro',
    'eucdov', 'eursta', 'fepowl', 'fiespa', 'flrtan1', 'foxspa', 'gadwal',
    'gamqua', 'gartro1', 'gbbgul', 'gbwwre1', 'gcrwar', 'gilwoo',
    'gnttow', 'gnwtea', 'gocfly1', 'gockin', 'gocspa', 'goftyr1',
    'gohque1', 'goowoo1', 'grasal1', 'grbani', 'grbher3', 'grcfly',
    'greegr', 'grekis', 'grepew', 'grethr1', 'gretin1', 'greyel',
    'grhcha1', 'grhowl', 'grnher', 'grnjay', 'grtgra', 'grycat',
    'gryhaw2', 'gwfgoo', 'haiwoo', 'heptan', 'hergul', 'herthr',
    'herwar', 'higmot1', 'hofwoo1', 'houfin', 'houspa', 'houwre',
    'hutvir', 'incdov', 'indbun', 'kebtou1', 'killde', 'labwoo', 'larspa',
    'laufal1', 'laugul', 'lazbun', 'leafly', 'leasan', 'lesgol', 'lesgre1',
    'lesvio1', 'linspa', 'linwoo1', 'littin1', 'lobdow', 'lobgna5', 'logshr',
    'lotduc', 'lotman1', 'lucwar', 'macwar', 'magwar', 'mallar3', 'marwre',
    'mastro1', 'meapar', 'melbla1', 'monoro1', 'mouchi', 'moudov', 'mouela1',
    'mouqua', 'mouwar', 'mutswa', 'naswar', 'norcar', 'norfli', 'normoc', 'norpar',
    'norsho', 'norwat', 'nrwswa', 'nutwoo', 'oaktit', 'obnthr1', 'ocbfly1',
    'oliwoo1', 'olsfly', 'orbeup1', 'orbspa1', 'orcpar', 'orcwar', 'orfpar',
    'osprey', 'ovenbi1', 'pabspi1', 'paltan1', 'palwar', 'pasfly', 'pavpig2',
    'phivir', 'pibgre', 'pilwoo', 'pinsis', 'pirfly1', 'plawre1', 'plaxen1',
    'plsvir', 'plupig2', 'prowar', 'purfin', 'purgal2', 'putfru1', 'pygnut',
    'rawwre1', 'rcatan1', 'rebnut', 'rebsap', 'rebwoo', 'redcro', 'reevir1',
    'rehbar1', 'relpar', 'reshaw', 'rethaw', 'rewbla', 'ribgul', 'rinkin1',
    'roahaw', 'robgro', 'rocpig', 'rotbec', 'royter1', 'rthhum', 'rtlhum',
    'ruboro1', 'rubpep1', 'rubrob', 'rubwre1', 'ruckin', 'rucspa1', 'rucwar',
    'rucwar1', 'rudpig', 'rudtur', 'rufhum', 'rugdov', 'rumfly1', 'runwre1',
    'rutjac1', 'saffin', 'sancra', 'sander', 'savspa', 'saypho', 'scamac1',
    'scatan', 'scbwre1', 'scptyr1', 'scrtan1', 'semplo', 'shicow', 'sibtan2',
    'sinwre1', 'sltred', 'smbani', 'snogoo', 'sobtyr1', 'socfly1', 'solsan',
    'sonspa', 'soulap1', 'sposan', 'spotow', 'spvear1', 'squcuc1', 'stbori',
    'stejay', 'sthant1', 'sthwoo1', 'strcuc1', 'strfly1', 'strsal1', 'stvhum2',
    'subfly', 'sumtan', 'swaspa', 'swathr', 'tenwar', 'thbeup1', 'thbkin',
    'thswar1', 'towsol', 'treswa', 'trogna1', 'trokin', 'tromoc', 'tropar',
    'tropew1', 'tuftit', 'tunswa', 'veery', 'verdin', 'vigswa', 'warvir',
    'wbwwre1', 'webwoo1', 'wegspa1', 'wesant1', 'wesblu', 'weskin', 'wesmea',
    'westan', 'wewpew', 'whbman1', 'whbnut', 'whcpar', 'whcsee1', 'whcspa',
    'whevir', 'whfpar1', 'whimbr', 'whiwre1', 'whtdov', 'whtspa', 'whwbec1',
    'whwdov', 'wilfly', 'willet1', 'wilsni1', 'wiltur', 'wlswar', 'wooduc',
    'woothr', 'wrenti', 'y00475', 'yebcha', 'yebela1', 'yebfly', 'yebori1',
    'yebsap', 'yebsee1', 'yefgra1', 'yegvir', 'yehbla', 'yehcar1', 'yelgro',
    'yelwar', 'yeofly1', 'yerwar', 'yeteup1', 'yetvir']

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

    # loss function, optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    # tensorboard config
    # writer = SummaryWriter(log_dir=LOG_DIR)

    # 一度callbackは省略!
    runner.train(
        model=model,
        criterion=loss_func,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=30,
        logdir=LOG_DIR,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,)

    """
    for epoch in range(epochs):
    # training
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print({'train/loss': epoch_loss})
            writer.add_scalar('train/loss', epoch_loss, global_step=epoch)

        # fix model for evaluation
        model.eval()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            epoch_loss += loss.item()
            print({'valid/loss': epoch_loss})
            writer.add_scalar('train/loss', epoch_loss, global_step=epoch)
    """
    
    # save model
    torch.save(model.state_dict(), 'model_gpu.pth')
    torch.save(model.state_dict().to('cpu'), 'model_cpu.pth')
