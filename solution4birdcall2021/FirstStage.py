import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
# from audiomentations import Compose, AddGausianNoise, PitchShift, 
import warnings

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from dataset import SpectrogramDataset1stStage, SpectrogramDataset2ndStage
from model import TimmSEDfromImage, TimmSEDfromSound
from loss import BCEFocal2WayLoss

warnings.simplefilter('ignore')

train_meta_path = '/home/yohei.nomoto/bird2021/solution4birdcall2021/data/small/train_metadata.csv'
train_meta = pd.read_csv(train_meta_path)

# define just 2 species
# train_meta = train_meta.query('primary_label in ["acafly", "acowoo"]')

BIRD_NAME = pd.read_csv('birdname.csv').columns.values
BIRD_CODE = {bird_name: i for i, bird_name in enumerate(BIRD_NAME)}
# BIRD_CODE = {'acafly': 0, 'acowoo': 1}

# validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_meta["fold"] = -1
for fold_id, (train_index, val_index) in enumerate(skf.split(train_meta, train_meta["primary_label"])):
    train_meta.iloc[val_index, -1] = fold_id

use_fold = 0
train_file_list = train_meta.query("fold != @use_fold")[['primary_label', 'filename']].values.tolist()
valid_file_list = train_meta.query("fold == @use_fold")[['primary_label', 'filename']].values.tolist()

class LitBirdcall2021(LightningModule):

    def __init__(self, data_dir='./'):
        super().__init__()
        self.data_dir = data_dir
        # TimmSED
        # input: [batch_size, time]
        # output: {framewise_output, segmentwise_output, logit, framewise_logit, clipwise_output}
        # 一時的にmodelのoutputのclassを2にする!
        self.num_classes = len(BIRD_NAME)
        self.model = TimmSEDfromImage(base_model_name='tf_efficientnet_b0_ns', pretrained=True, num_classes=self.num_classes, in_channels=1)
        self.loss_func = BCEFocal2WayLoss()

        self.trainset = SpectrogramDataset1stStage(file_list=train_file_list)
        self.valset = SpectrogramDataset1stStage(file_list=valid_file_list)

    def forward(self, x):
        output = self.model(x)
        return output

    def train_dataloader(self):
        train_dl = DataLoader(self.trainset, batch_size=128, shuffle=True, num_workers=4)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.valset, batch_size=128, num_workers=4)
        return val_dl
    
    def loss_function(self, preds, labels):
        loss = self.loss_func(preds, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        signal, targets = batch
        preds = self.model(signal)
        loss = self.loss_function(preds, targets)
        outputs_, y_ = preds['clipwise_output'].cpu().detach().numpy(), targets.cpu().detach().numpy()
        F1score_3 = f1_score(y_, outputs_ > 0.3, average='samples')
        F1score_5 = f1_score(y_, outputs_ > 0.5, average='samples')
        F1score_7 = f1_score(y_, outputs_ > 0.7, average='samples')
        mAPscore = average_precision_score(y_, outputs_, average=None)
        mAPscore = np.nan_to_num(mAPscore).mean()
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1_0.3_step', F1score_3, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1_0.5_step', F1score_5, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1_0.7_step', F1score_7, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mAP_step', mAPscore, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        signal, targets = batch
        preds = self.model(signal)
        return {'output': preds, 'y': targets}

    # まず, f1やmAPの計算方法を再確認する
    def validation_epoch_end(self, outputs):
        clipwise_output, y = [], []
        loss = 0.
        for output in outputs:
            loss += self.loss_function(output['output'], output['y']).item()
            clipwise_output.append(output['output']['clipwise_output'].cpu().detach().numpy())
            y.append(output['y'].cpu().detach().numpy())
        clipwise_output, y = np.array(clipwise_output).reshape(-1, self.num_classes), np.array(y).reshape(-1, self.num_classes)
        F1score_3 = f1_score(y, clipwise_output > 0.3, average='samples')
        F1score_5 = f1_score(y, clipwise_output > 0.5, average='samples')
        F1score_7 = f1_score(y, clipwise_output > 0.7, average='samples')
        mAPscore = average_precision_score(y, clipwise_output, average=None)
        mAPscore = np.nan_to_num(mAPscore).mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_0.3', F1score_3, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_0.5', F1score_5, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_0.7', F1score_7, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mAP', mAPscore, on_epoch=True, prog_bar=True, logger=True)
        return loss

if __name__ == '__main__':
    seed_everything(42, workers=True)
    model = LitBirdcall2021()
    checkpoint_cb = ModelCheckpoint(dirpath='./reports/', filename='model_weight', save_weights_only=True, monitor="val_loss", mode="min", save_last=True)
    earlystop_cb = EarlyStopping(monitor="val_loss", mode="min")
    trainer = Trainer(gpus=1, max_epochs=2, deterministic=True, callbacks=[checkpoint_cb, earlystop_cb])
    trainer.fit(model)
