import torch
import torch.nn as nn

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, preds, targets):
        # BCEWithLogitsLoss: Binary Cross Entropyにsigmoidを重ねることで安定化したもの
        # BCE loss
        # loss(o,t)=-1/n * \Sigma_{i}(t[i]log(o[i]) + (1 - t[i]log(1-o[i])))
        # BCEWithLogitsLoss
        # loss(o,t)=-1/n * \Sigma_{i}(t[i]log(sigmoid(o[i])) + (1 - t[i]log(1-sigmoid(o[i]))))
        # preds, targets: [batch_size, num_classes]
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1.-probas) ** self.gamma + (1.-targets) * probas ** self.gamma
        loss *= bce_loss
        loss = loss.mean()
        return loss

# 音データ全体によるlogitと, frameごとに切れたlogitの両方を用いてlossを計算する
class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1,1], class_weights=None):
        super().__init__()
        
        self.focal = BCEFocalLoss()
        self.weights = weights
        
    def forward(self, input, target):
        # input: modelの吐き出したpreds
        clipwise_preds = input['logit']
        target = target.float()
        framewise_preds = input['framewise_logit']
        # torchのmaxはindicesを返すので, _で消去
        clipwise_preds_from_framewise_preds, _ = framewise_preds.max(dim=1)
        
        loss = self.focal(clipwise_preds, target)
        aux_loss = self.focal(clipwise_preds_from_framewise_preds, target)
        
        return self.weights[0] * loss + self.weights[1] * aux_loss