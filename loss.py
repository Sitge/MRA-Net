# Dice-BCE loss

import torch
import torch.nn as nn


# Dice-BCE loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class dice_bce_mae_loss(nn.Module):
    def __init__(self, batch=True, lambda_mae=0.5):
        super(dice_bce_mae_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.lambda_mae = lambda_mae

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.01
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        return 1 - self.soft_dice_coeff(y_true, y_pred)

    def __call__(self, y_true, y_pred):
        bce = self.bce_loss(y_pred, y_true)
        dice = self.soft_dice_loss(y_true, y_pred)
        mae = F.l1_loss(y_pred, y_true)
        return bce + dice + self.lambda_mae * mae
