'''Module to add some Loss functions that are not implemented in Pytorth'''
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() +
                                           targets.sum() +
                                           smooth)

        return 1 - dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predicted, target):
        predicted = predicted.view(-1)
        target = target.view(-1)

        true_positives = torch.sum(predicted * target)
        false_positives = torch.sum(predicted * (1 - target))
        false_negatives = torch.sum((1 - predicted) * target)

        tversky_index = true_positives / (true_positives +
                                          self.alpha * false_positives +
                                          self.beta * false_negatives)
        tversky_loss = 1 - tversky_index

        return tversky_loss
