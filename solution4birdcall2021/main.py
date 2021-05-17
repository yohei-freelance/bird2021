import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score
import warnings

import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from dataset import SpectrogramDataset
from model import TimmSED
from loss import BCEFocal2WayLoss

warnings.simplefilter('ignore')

train_meta_path = '/home/yohei.nomoto/bird2021/solution4birdcall2021/data/small/train_metadata.csv'
train_meta = pd.read_csv(train_meta_path)

# define just 2 species
train_meta = train_meta.query('primary_label in ["acafly", "acowoo"]')
# BIRD_CODE = {bird_name: i for i, bird_name in enumerate(target_columns)}
BIRD_CODE = {'acafly': 0, 'acowoo': 1}

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
        self.model = TimmSED(base_model_name='tf_efficientnet_b0_ns', pretrained=True, num_classes=2, in_channels=1)
        self.loss_func = BCEFocal2WayLoss()

        self.trainset = SpectrogramDataset(file_list=train_file_list)
        self.valset = SpectrogramDataset(file_list=valid_file_list)

    def forward(self, x):
        output = self.model(x)
        return output

    def train_dataloader(self):
        train_dl = DataLoader(self.trainset, batch_size=5, shuffle=True, num_workers=2)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.valset, batch_size=5, num_workers=2)
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
        outputs_, y_ = preds['clipwise_output'].numpy(), targets.numpy()
        F1score = f1score(outputs_, y_)
        mAPscore = average_precision_score(outputs_, y_)
        mAPscore = np.nan_to_num(mAPscore).mean()
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1_step', F1score, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mAP_step', mAPscore, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        signal, targets = batch
        preds = self.model(signal)
        return {'output': preds, 'y': targets}

    def validation_epoch_end(self, outputs):
        outputs_, y_ = [], []
        for output in outputs:
            outputs_.append(output['output'])
            y_.append(output['y'])
        outputs_ = torch.cat(outputs_)
        y_ = torch.cat(y_)
        loss = self.loss_function(outputs_, y_)
        outputs_, y_ = outputs_['clipwise_output'].numpy(), y_.numpy()
        F1score = f1score(outputs_, y_)
        mAPscore = average_precision_score(outputs_, y_)
        mAPscore = np.nan_to_num(mAPscore).mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', F1score, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mAP', mAPscore, on_epoch=True, prog_bar=True, logger=True)
        return loss

if __name__ == '__main__':
    seed_everything(42, workers=True)
    model = LitBirdcall2021()
    checkpoint_cb = ModelCheckpoint(dirpath='./reports/', filename=f'epoch-{epoch}--val_loss-{val_loss}--val_f1-{val_f1}', save_weights_only=True, monitor="val_loss", mode="min", save_last=True)
    earlystop_cb = EarlyStopping(monitor="val_loss", mode="min")
    trainer = Trainer(gpus=1, max_epochs=1, deterministic=True, callbacks=[checkpoint_cb, earlystop_cb])
    trainer.fit(model)
