import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion:

    def __init__(self):
        pass

    @staticmethod
    def adversarial(pos_preds, neg_preds, neg_rate, alpha):
        # RotatE: Adversarial Negative Sampling and alpha is the temperature.
        pos_preds = -pos_preds
        neg_preds = -neg_preds
        pos_preds = F.logsigmoid(pos_preds)
        neg_preds = neg_preds.view((-1, neg_rate))
        softmax = nn.Softmax(dim=1)(neg_preds * alpha).detach()
        neg_preds = torch.sum(softmax * (F.logsigmoid(-neg_preds)), dim=-1)
        loss = -neg_preds.mean() - pos_preds.mean()
        return loss

    @staticmethod
    def margin(pos_preds, neg_preds, margin):
        loss = pos_preds + margin - neg_preds
        loss = torch.max(loss, torch.zeros_like(loss)).sum()
        return loss

    @staticmethod
    def dnn(pred_heads, pred_tails, tr_h, hr_t, label_smoothing, tot_entity):
        hr_t = hr_t * (1.0 - label_smoothing) + 1.0 / tot_entity
        tr_h = tr_h * (1.0 - label_smoothing) + 1.0 / tot_entity
        loss_heads = torch.mean(torch.nn.BCEWithLogitsLoss()(pred_heads, tr_h))
        loss_tails = torch.mean(torch.nn.BCEWithLogitsLoss()(pred_tails, hr_t))
        loss = loss_heads + loss_tails
        return loss

    @staticmethod
    def sum(pred_heads, pred_tails):
        loss = pred_heads + pred_tails
        return loss

    @staticmethod
    def pointwise(preds, target):
        loss = F.softplus(target*preds).mean()
        return loss

    @staticmethod
    def hyperbolic(preds, target):
        loss = torch.nn.BCEWithLogitsLoss()(preds, target)
        return loss