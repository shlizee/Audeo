import Video2RollNet
import os
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from Video2Roll_dataset import Video2RollDataset
from torch.utils.data import DataLoader
import torch
import time
from sklearn import metrics
from sklearn.metrics import _classification
import torch.nn as nn
def validate(net, criterion, test_loader):
    epoch_loss = 0
    count = 0
    all_pred_label = []
    all_label = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            imgs, label = data
            logits = net(imgs)
            loss = criterion(logits, label)
            pred_label = torch.sigmoid(logits) >= 0.4
            numpy_label = label.cpu().detach().numpy().astype(np.int)
            numpy_pre_label = pred_label.cpu().detach().numpy().astype(np.int)
            all_label.append(numpy_label)
            all_pred_label.append(numpy_pre_label)
            epoch_loss += loss.item()
            count += 1
    all_label = np.vstack(all_label)
    all_pred_label = np.vstack(all_pred_label)
    labels = _classification._check_set_wise_labels(all_label, all_pred_label,labels=None, pos_label=1, average='samples')
    MCM = metrics.multilabel_confusion_matrix(all_label, all_pred_label,sample_weight=None, labels=labels, samplewise=True)
    tp_sum = MCM[:, 1, 1]
    fp_sum = MCM[:, 0, 1]
    fn_sum = MCM[:, 1, 0]
    # tn_sum = MCM[:, 0, 0]
    accuracy = _prf_divide(tp_sum, tp_sum+fp_sum+fn_sum, zero_division=1)
    accuracy = np.average(accuracy)
    all_precision = metrics.precision_score(all_label, all_pred_label, average='samples', zero_division=1)
    all_recall = metrics.recall_score(all_label, all_pred_label, average='samples', zero_division=1)
    all_f1_score = metrics.f1_score(all_label, all_pred_label, average='samples', zero_division=1)
    return epoch_loss/count, all_precision, all_recall, accuracy, all_f1_score


def _prf_divide(numerator, denominator, zero_division="warn"):
    """Performs division and handles divide-by-zero.
    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.
    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ["warn", 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != "warn":
        return result

if __name__ == "__main__":
    model_path = './models/BCE_5f_FAN.pth'
    device = torch.device('cuda')
    net = Video2RollNet.resnet18()
    # net = torch.nn.DataParallel(net)
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    print(net)
    test_dataset = Video2RollDataset(subset='test')
    test_data_loader = DataLoader(test_dataset, batch_size=64)
    net.eval()
    criterion=nn.BCEWithLogitsLoss()
    val_avg_loss, val_avg_precision, val_avg_recall, val_avg_acc, val_fscore = validate(net, criterion, test_data_loader)
    epoch = 0
    print('-' * 85)
    print(
        "epoch {0} validation loss:{1:.3f} | avg precision:{2:.3f} | avg recall:{3:.3f} | avg acc:{4:.3f} | f1 score:{5:.3f}".format(
            epoch + 1, val_avg_loss, val_avg_precision, val_avg_recall, val_avg_acc, val_fscore))
    print('-' * 85)