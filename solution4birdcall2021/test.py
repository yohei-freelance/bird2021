import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
# from audiomentations import Compose, AddGausianNoise, PitchShift, 
import warnings

import torch
from torch.utils.data import DataLoader
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from dataset import SpectrogramDataset1stStage, SpectrogramDataset2ndStage
from model import TimmSEDfromImage, TimmSEDfromSound
from loss import BCEFocal2WayLoss

warnings.simplefilter('ignore')

test_path = '/Users/yohei/Documents/bird2021/solution4birdcall2021/data/small/test.csv'
test = pd.read_csv(test_path)
weight_path = '/Users/yohei/Documents/bird2021/solution4birdcall2021/lightning_logs/reports/first-stage--epoch=05-val_loss=0.27.ckpt'

BIRD_NAME = pd.read_csv('birdname.csv').columns.values
BIRD_CODE = {bird_name: i for i, bird_name in enumerate(BIRD_NAME)}

class LitBirdcall2021(LightningModule):

    def __init__(self, data_dir='./'):
        super().__init__()
        self.data_dir = data_dir
        # TimmSED
        # input: [batch_size, time]
        # output: {framewise_output, segmentwise_output, logit, framewise_logit, clipwise_output}
        self.num_classes = len(BIRD_NAME)
        self.model = TimmSEDfromImage(base_model_name='tf_efficientnet_b0_ns', pretrained=False, num_classes=self.num_classes, in_channels=1)
        # GPU上でしか読み出せない
        self.model.load_state_dict(torch.load(weight_path))
        self.spectrogram_extractor = Spectrogram(n_fft=2048, hop_length=512, win_length=2048, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=2048, n_mels=128, fmin=20, fmax=16000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.loss_func = BCEFocal2WayLoss()

        self.testset = TestDataset()
        self.threshold = 0.5

    def forward(self, x):
        x = self.spectogram_extractor(x)
        x = self.logmel_extractor(x)
        # この時点で0~1のレンジになっているかチェックが必要.
        x = x.transpose(2, 3)
        output = self.model(x)
        return output

    def test_dataloader(self):
        test_dl = DataLoader(self.testset, batch_size=1, shuffle=False)
        return test_dl
    
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

    def test_step(self, batch, batch_idx):
        for signal, row_id in tqdm(batch):
            row_id = row_id[0]

            # with torch.no_grad():
            prediction = self.forward(signal)
            proba = prediction["clipwise_output"].detach().cpu().numpy().reshape(-1)

            events = proba >= self.threshold
            labels = np.argwhere(events).reshape(-1).tolist()

            if len(labels) == 0:
                prediction_dict[row_id] = "nocall"
            else:
                labels_str_list = list(map(lambda x: BIRD_CODE[x], labels))
                label_string = " ".join(labels_str_list)
                prediction_dict[row_id] = label_string
        return prediction_dict

    # まず, f1やmAPの計算方法を再確認する
    def validation_epoch_end(self, outputs):
        loss = 0.
        for i, output in enumerate(outputs):
            try:
                clipwise_output = np.append(clipwise_output, output['output']['clipwise_output'].cpu().detach().numpy(), axis=0)
                y = np.append(y, output['y'].cpu().detach().numpy(), axis=0)
            except UnboundLocalError:
                clipwise_output = output['output']['clipwise_output'].cpu().detach().numpy()
                y = output['y'].cpu().detach().numpy()
            loss += self.loss_function(output['output'], output['y']).item()
        clipwise_output = clipwise_output.reshape(-1, self.num_classes)
        y = np.array(y).reshape(-1, self.num_classes)
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
    checkpoint_cb = ModelCheckpoint(dirpath='./reports/', filename='first-stage--{epoch:02d}-{val_loss:.2f}', save_weights_only=True, monitor="val_loss", mode="min", save_last=True)
    earlystop_cb = EarlyStopping(monitor="val_loss", mode="min")
    trainer = Trainer(gpus=1, max_epochs=30, deterministic=True, callbacks=[checkpoint_cb, earlystop_cb], precision=16)
    trainer.fit(model)
    trainer.save_checkpoint("final_model.ckpt")
    print(checkpoint_cb.best_model_path)
