# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BlurryLoss(nn.Module):
    def __init__(self, gamma: float = 0.0, cutoff_pt: float = 0.0, alpha=None, size_average: bool = True):
        super(BlurryLoss, self).__init__()
        self.gamma = float(gamma)
        self.cutoff_pt = cutoff_pt
        self.alpha = None
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.Tensor([alpha, 1 - alpha])
            elif isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target).view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        mask = pt >= self.cutoff_pt
        loss = - (pt ** self.gamma) * logpt
        loss = loss * mask.float()
        return loss.mean() if self.size_average else loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, cutoff_pt: float = 0.0, alpha=None, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = float(gamma)
        self.cutoff_pt = cutoff_pt
        self.alpha = None
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.Tensor([alpha, 1 - alpha])
            elif isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target).view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = - ((1 - pt) ** self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()

class GCELoss(nn.Module):
    def __init__(self, q: float = 0.7, ignore_index: int = -100):
        super(GCELoss, self).__init__()
        self.q = q
        self.ignore_index = ignore_index
        self.eps = 1e-8

    def forward(self, logits, targets):
        valid_idx = targets != self.ignore_index
        logits = logits[valid_idx]
        targets = targets[valid_idx]
        if self.q == 0:
            if logits.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(logits.view(-1), targets.float())
            else:
                ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
                loss = ce_loss(logits, targets)
            loss = (loss.view(-1) + self.eps).mean()
        else:
            if logits.size(-1) == 1:
                pred = torch.sigmoid(logits)
                pred = torch.cat((1 - pred, pred), dim=-1)
            else:
                pred = torch.softmax(logits, dim=-1)
            pred = torch.clamp(pred, min=self.eps, max=1.0)
            numerator = 1 - (pred ** self.q)
            denominator = self.q
            loss = numerator / (denominator + self.eps)
            target_one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.size(-1)).float()
            loss = torch.sum(loss * target_one_hot, dim=-1)
            loss = loss.mean()
        return loss

class NormalizedNegativeCrossEntropy(nn.Module):
    def __init__(self, num_classes: int, min_prob: float):
        super().__init__()
        self.num_classes = num_classes
        self.min_prob = min_prob
        self.A = - torch.tensor(min_prob).log()

    def forward(self, pred, labels):
        pred = torch.softmax(pred, dim=1).clamp(min=self.min_prob, max=1)
        pred = self.A + pred.log()
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()
