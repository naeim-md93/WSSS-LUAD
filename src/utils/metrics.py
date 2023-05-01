import numpy as np
from torch import nn
import torch.nn.functional as F


class ClassificationEvaluator:
    """
                    Target
                    0   1
                0   TN  FN
        Probs   1   FP  TP
    """
    def __init__(self, num_classes=4, thresh=0.5, epsilon=1e-9):
        self.thresh = thresh
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.cms = [np.zeros(shape=(2, 2)) for _ in range(self.num_classes)]

    def reset_cms(self):
        self.cms = [np.zeros(shape=(2, 2)) for _ in range(self.num_classes)]

    def get_cms(self):
        return self.cms

    def update_cms(self, pred_labels, target):
        for b in range(pred_labels.shape[0]):
            for c in range(pred_labels.shape[1]):
                self.cms[c][pred_labels[b, c], target[b, c]] += 1

    def calc_accuracy(self, correct_1_labels, pred_labels, target):
        acc_intersection = correct_1_labels.sum(axis=1)
        acc_union = target.sum(axis=1) + pred_labels.sum(axis=1) - acc_intersection
        acc = np.mean(a=(acc_intersection / acc_union))
        return acc

    def calc_class_accuracy(self, correct_1_labels, pred_labels, target):
        class_intersection = correct_1_labels.sum(axis=0)
        class_union = target.sum(axis=0) + pred_labels.sum(axis=0) - class_intersection
        class_acc = class_intersection / (class_union + self.epsilon)
        return class_acc

    def __call__(self, probs, target):
        pred_labels = (probs > self.thresh).astype(np.uint8)

        self.update_cms(pred_labels=pred_labels, target=target)

        correct_1_labels = pred_labels * target

        return (
            # Accuracy
            self.calc_accuracy(correct_1_labels=correct_1_labels, pred_labels=pred_labels, target=target),
            # Exact Match
            np.mean(a=(pred_labels == target).sum(axis=1) == self.num_classes),
            # Class Accuracy
            self.calc_class_accuracy(correct_1_labels=correct_1_labels, pred_labels=pred_labels, target=target),
            # TE Confusion Matrix
            self.cms[0],
            # NEC Confusion Matrix
            self.cms[1],
            # LYM Confusion Matrix
            self.cms[2],
            # TAS Confusion Matrix
            self.cms[3],
        )


class WMLSMLoss:
    """
    Weighted Multi-Label Soft Margin Loss
    """
    def __init__(self, tw, fw):
        """

        :param tw: true (1) weight
        :param fw: false (0) weight
        """
        super(WMLSMLoss, self).__init__()
        self.tw = tw
        self.fw = fw

    def train_call(self, logits, target):

        true_loss = self.tw * target * F.logsigmoid(input=logits)
        false_loss = self.fw * (1 - target) * F.logsigmoid(input=-logits)

        loss = -(true_loss + false_loss)

        if target.ndim == 1:
            return loss
        elif target.ndim == 2:
            return loss.sum(dim=0)
        else:
            raise NotImplementedError('Not implemented yet')

    def val_call(self, logits, target):

        loss = -(target * F.logsigmoid(logits) + (1 - target) * F.logsigmoid(-logits))

        if target.ndim == 1:
            return loss
        elif target.ndim == 2:
            return loss.sum(dim=0)
        else:
            raise NotImplementedError('Not implemented yet')