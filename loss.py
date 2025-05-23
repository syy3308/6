import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self, conf_weight=1.0, loc_weight=1.0):
        super().__init__()
        self.conf_weight = conf_weight
        self.loc_weight = loc_weight

    def forward(self, conf_preds, loc_preds, conf_targets, loc_targets):
        # 分类损失
        conf_loss = F.cross_entropy(conf_preds, conf_targets.long())

        # 定位损失
        loc_loss = F.smooth_l1_loss(loc_preds, loc_targets)

        # 总损失
        total_loss = self.conf_weight * conf_loss + self.loc_weight * loc_loss

        return total_loss
