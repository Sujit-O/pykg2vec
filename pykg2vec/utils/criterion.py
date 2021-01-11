import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion:
    """Utility for calculating KGE losses

       Loss Functions in Knowledge Graph Embedding Models
       http://ceur-ws.org/Vol-2377/paper_1.pdf
    """

    @staticmethod
    def pariwise_logistic(pos_preds, neg_preds, neg_rate, alpha):
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
    def pairwise_hinge(pos_preds, neg_preds, margin):
        loss = pos_preds + margin - neg_preds
        loss = torch.max(loss, torch.zeros_like(loss)).sum()
        return loss

    @staticmethod
    def pointwise_logistic(preds, target):
        loss = F.softplus(target*preds).mean()
        return loss

    @staticmethod
    def pointwise_bce(preds, target):
        loss = torch.nn.BCEWithLogitsLoss()(preds, torch.clamp(target, min=0.0, max=1.0))
        return loss

    @staticmethod
    def multi_class_bce(pred_heads, pred_tails, tr_h, hr_t, label_smoothing, tot_entity):
        if label_smoothing is not None and tot_entity is not None:
            hr_t = hr_t * (1.0 - label_smoothing) + 1.0 / tot_entity
            tr_h = tr_h * (1.0 - label_smoothing) + 1.0 / tot_entity
        loss_heads = torch.mean(torch.nn.BCEWithLogitsLoss()(pred_heads, tr_h))
        loss_tails = torch.mean(torch.nn.BCEWithLogitsLoss()(pred_tails, hr_t))
        loss = loss_heads + loss_tails
        return loss

    @staticmethod
    def multi_class(pred_heads, pred_tails):
        loss = pred_heads + pred_tails
        return loss
