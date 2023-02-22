import numpy as np
import torch
import torch.nn.functional as F


class IoUAccuracy:
    def __init__(self, num_classes=4, thresh=0.5, epsilon=1e-9):
        self.thresh = thresh
        self.num_classes = num_classes
        self.epsilon = epsilon

    def __call__(self, probs, y):
        pred_labels = (probs >= self.thresh) * 1
        correct_labels = (pred_labels == y) * 1
        correct_1_labels = pred_labels * y

        acc_intersection = correct_1_labels.sum(axis=1)
        acc_union = y.sum(axis=1) + pred_labels.sum(axis=1) - acc_intersection
        acc = np.mean(a=(acc_intersection / acc_union))

        em = np.mean(a=correct_labels.sum(axis=1) == self.num_classes)

        class_intersection = correct_1_labels.sum(axis=0)
        class_union = y.sum(axis=0) + pred_labels.sum(axis=0) - class_intersection
        class_acc = class_intersection / (class_union + self.epsilon)

        return {'accuracy': acc, 'exact_match': em, 'class_acc': class_acc}


class PseudoMaskEvaluator:
    def __init__(self, num_classes, epsilon=1e-12):
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes))

    def add_batch(self, gt_mask, pred_mask):
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()
        self.confusion_matrix += self.get_confusion_matrix(gt_mask=gt_mask, pred_mask=pred_mask)

    def reset(self):
        self.confusion_matrix = np.zeros(shape=(self.num_classes, self.num_classes))

    def get_confusion_matrix(self, gt_mask, pred_mask):  # (N, 224, 224)
        mask = (gt_mask >= 0) & (gt_mask < self.num_classes)
        label = (self.num_classes * gt_mask[mask].astype(np.uint8)) + pred_mask[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape((self.num_classes, self.num_classes))
        return confusion_matrix

#     def get_class_scores(self, cm, target):
#         class_cm = np.zeros(shape=(2, 2))
#         TP = cm[target, target]
#         FP = np.sum(a=cm[:, target]) - TP
#         FN = np.sum(a=cm[target, :]) - TP
#         TN = np.sum(a=cm) - (TP + FP + FN)
#
#         class_cm[0, 0] = TP
#         class_cm[0, 1] = FN
#         class_cm[1, 0] = FP
#         class_cm[1, 1] = TN
#
#         precision = TP / (TP + FP + self.epsilon)  # PPV
#         recall = TP / (TP + FN + self.epsilon)  # Sensitivity
#
#         class_scores = {
#             'recall': recall,  # Sensitivity
#             'specificity': TN / (TN + FP + self.epsilon),
#             'precision': precision,  # PPV
#             'npv': TN / (TN + FN + self.epsilon),
#             'f1': (2 * precision * recall) / (precision + recall + self.epsilon),
#             'iou': TP / (TP + FN + FP + self.epsilon),
#             'accuracy': (TP + TN) / (TP + TN + FP + FN + self.epsilon)
#         }
#
#         return class_scores

    def get_scores(self):  # (b, 5, 224, 224)

        diag = np.diag(v=self.confusion_matrix)
        sum_0 = self.confusion_matrix.sum(axis=0)
        sum_1 = self.confusion_matrix.sum(axis=1)

        iou = diag / (sum_1 + sum_0 - diag)
        freqs = sum_1 / self.confusion_matrix.sum()

        res = {
            'pa': diag.sum() / self.confusion_matrix.sum(),  # Pixel Accuracy
            'ma': np.nanmean(a=(diag / sum_1)),  # Mean Accuracy / Pixel Accuracy Class
            'iou': iou,  # Intersection Over Union
            'miou': np.nanmean(a=iou),  # Mean IoU
            'fwiou': (freqs[freqs > 0] * iou[freqs > 0]).sum(),  # FW IoU
        }

        return res


class WeightedMultiLabelSoftMarginLoss():
    def __init__(self, tw=1, nw=1):
        super(WeightedMultiLabelSoftMarginLoss, self).__init__()
        self.tw = tw
        self.nw = nw

    def train_call(self, input, target):

        tmp = target.clone().detach()

        if len(tmp.size()) == 1:
            num_classes = tmp.size(0)
            true_sum = (tmp == 1) * 1
            false_sum = (target == 0) * 1
        else:
            num_classes = tmp.size(1)
            true_sum = (tmp == 1).sum(dim=0)
            false_sum = (target == 0).sum(dim=0)

        true_weights = ((num_classes + 1) - true_sum) * self.tw
        false_weights = ((num_classes + 1) - false_sum) * self.nw

        pos = target * F.logsigmoid(input) * true_weights
        neg = (1 - target) * F.logsigmoid(-input) * false_weights
        loss = -(pos + neg)

        return loss.mean(dim=0)

    def val_call(self, input, target):
        loss = -((target * F.logsigmoid(input)) + ((1 - target) * F.logsigmoid(-input)))
        return loss.mean(dim=0)
