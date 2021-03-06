import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import timm

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output

# weight initialization module
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

# attention module
class AttBlockV2(nn.Module):
    def __init__ (self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
    
        # weight initialization
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        # n_samples: batch_size, n_in: basemodel->fully connected??????channel???
        # n_out: ????????????????????????
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        # norm_att: (n_samples, n_out, n_time) n_time??????softmax???????????????????????????
        cla = self.nonlinear_transform(self.cla(x))
        # cla: (n_samples, n_out, n_time)
        x = torch.sum(norm_att * cla, dim=2)
        # x: (n_samples, n_out)
        return x, norm_att, cla

# sound event detection module
# numpy based input or not
class TimmSEDfromImage(nn.Module):

    # base_model_name: you can select arbitary model from https://github.com/rwightman/pytorch-image-models/tree/master/timm/models

    def __init__(self, base_model_name: str, num_classes: int, pretrained=False, in_channels=1):
        super().__init__()

        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(128)

        # main module
        self.base_model_name = base_model_name
        base_model = timm.create_model(self.base_model_name, pretrained=True, in_chans=1)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        
        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features
        
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(in_features, num_classes, activation="sigmoid")

        # weight initialization
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)

    def forward(self, input):
        x = input.transpose(2, 3)
        # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]
        
        # mel_bins????????????????????????????????????
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # augmentation
        if self.training:
            x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        # (batch_size, 1, mel_bins(normalized), time_steps)
        x = self.encoder(x)
        # (batch_size, channels, freq, frames)
        x = torch.mean(x, dim=2)
        # (batch_size, channels, frames)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        # (batch_size, frames, channels)
        # ????????????????????????????????????????????????
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        # (batch_size, channels, frames)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        # clipwise_output: (batch_size, class_num)
        # norm_att: (batch_size, class_num, frames), frames?????????????????????
        # segmentwise_output: (batch_size, class_num, frames)

        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        # logit: (batch_size, class_num)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        # segmentwise_logit: (batch_size, frames, class_num)
        segmentwise_output = segmentwise_output.transpose(1, 2)
        # segmentwise_output: (batch_size, frames, class_num)

        interpolate_ratio = frames_num // segmentwise_output.size(1)
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict


class TimmSEDfromSound(TimmSEDfromImage):
    # base_model_name: you can select arbitary model from https://github.com/rwightman/pytorch-image-models/tree/master/timm/models

    def __init__(self, base_model_name: str, num_classes: int, pretrained=False, in_channels=1):
        super().__init__(base_model_name=base_model_name, pretrained=pretrained, num_classes=num_classes, in_channels=in_channels)

        # spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=2048, hop_length=512, win_length=2048, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        # logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=2048, n_mels=128, fmin=20, fmax=16000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

    def forward(self, input):
        # (batch_size, data_length)
        x = self.spectrogram_extractor(input)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)
        x = self.logmel_extractor(x)
        # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]
        
        # mel_bins????????????????????????????????????
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # augmentation
        if self.training:
            x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        # (batch_size, 1, mel_bins(normalized), time_steps)
        x = self.encoder(x)
        # (batch_size, channels, freq, frames)
        x = torch.mean(x, dim=2)
        # (batch_size, channels, frames)

        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        # (batch_size, frames, channels)
        # ????????????????????????????????????????????????
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        # (batch_size, channels, frames)
        x = F.dropout(x, p=0.5, training=self.training)
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        # clipwise_output: (batch_size, class_num)
        # norm_att: (batch_size, class_num, frames), frames?????????????????????
        # segmentwise_output: (batch_size, class_num, frames)

        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        # logit: (batch_size, class_num)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        # segmentwise_logit: (batch_size, frames, class_num)
        segmentwise_output = segmentwise_output.transpose(1, 2)
        # segmentwise_output: (batch_size, frames, class_num)

        interpolate_ratio = frames_num // segmentwise_output.size(1)
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
            "clipwise_output": clipwise_output
        }

        return output_dict