import os
import json
from Roll2Midi_dataset import Roll2MidiDataset
from sklearn import metrics
import torch.utils.data as utils
import torch
from Roll2MidiNet import Generator
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import _classification
cuda = torch.device("cuda")
Tensor = torch.cuda.FloatTensor
def process_data():
    test_dataset = Roll2MidiDataset(train=False)
    test_loader = utils.DataLoader(test_dataset, batch_size=16)
    return test_loader

def test(generator,  test_loader):
    all_label = []
    all_pred_label = []
    all_pred_label_ = []
    with torch.no_grad():
        generator.eval()
        for idx, data in enumerate(test_loader):
            gt, roll = data
            # Adversarial ground truths
            gt = gt.type(Tensor)
            roll = roll.type(Tensor)

            real = Variable(gt)
            roll_ = Variable(roll)
            gen_imgs = generator(roll_)

            pred_label = gen_imgs >= 0.4
            numpy_label = gt.cpu().detach().numpy().astype(np.int) # B,1, 51, 50
            numpy_label = np.transpose(numpy_label.squeeze(), (0, 2, 1))  # B,50,51
            numpy_label = np.reshape(numpy_label, (-1, 51))
            numpy_pre_label = pred_label.cpu().detach().numpy().astype(np.int)
            numpy_pre_label = np.transpose(numpy_pre_label.squeeze(), (0, 2, 1)) #B,50,51
            numpy_pre_label = np.reshape(numpy_pre_label, (-1, 51))
            all_label.append(numpy_label)
            all_pred_label.append(numpy_pre_label)

            pred_label_ = gen_imgs >= 0.5
            numpy_pre_label_ = pred_label_.cpu().detach().numpy().astype(np.int)
            numpy_pre_label_ = np.transpose(numpy_pre_label_.squeeze(), (0, 2, 1))  # B,50,51
            numpy_pre_label_ = np.reshape(numpy_pre_label_, (-1, 51))
            all_pred_label_.append(numpy_pre_label_)

        all_label = np.vstack(all_label)
        all_pred_label = np.vstack(all_pred_label)
        labels = _classification._check_set_wise_labels(all_label, all_pred_label, labels=None, pos_label=1,
                                                        average='samples')
        MCM = metrics.multilabel_confusion_matrix(all_label, all_pred_label, sample_weight=None, labels=labels,
                                                  samplewise=True)
        tp_sum = MCM[:, 1, 1]
        fp_sum = MCM[:, 0, 1]
        fn_sum = MCM[:, 1, 0]
        # tn_sum = MCM[:, 0, 0]
        accuracy = _prf_divide(tp_sum, tp_sum + fp_sum + fn_sum, zero_division=1)
        accuracy = np.average(accuracy)
        all_precision = metrics.precision_score(all_label, all_pred_label, average='samples', zero_division=1)
        all_recall = metrics.recall_score(all_label, all_pred_label, average='samples', zero_division=1)
        all_f1_score = metrics.f1_score(all_label, all_pred_label, average='samples', zero_division=1)
        print(
            "Threshold 0.4, avg precision:{0:.3f} | avg recall:{1:.3f} | avg acc:{2:.3f} | f1 score:{3:.3f}".format(
                 all_precision, all_recall, accuracy, all_f1_score))

        all_pred_label_ = np.vstack(all_pred_label_)
        labels = _classification._check_set_wise_labels(all_label, all_pred_label_, labels=None, pos_label=1,
                                                        average='samples')
        MCM = metrics.multilabel_confusion_matrix(all_label, all_pred_label_, sample_weight=None, labels=labels,
                                                  samplewise=True)
        tp_sum = MCM[:, 1, 1]
        fp_sum = MCM[:, 0, 1]
        fn_sum = MCM[:, 1, 0]
        # tn_sum = MCM[:, 0, 0]
        accuracy = _prf_divide(tp_sum, tp_sum + fp_sum + fn_sum, zero_division=1)
        accuracy = np.average(accuracy)
        all_precision = metrics.precision_score(all_label, all_pred_label_, average='samples', zero_division=1)
        all_recall = metrics.recall_score(all_label, all_pred_label_, average='samples', zero_division=1)
        all_f1_score = metrics.f1_score(all_label, all_pred_label_, average='samples', zero_division=1)
        print(
            "Threshold 0.5,  avg precision:{0:.3f} | avg recall:{1:.3f} | avg acc:{2:.3f} | f1 score:{3:.3f}".format(
                all_precision, all_recall,accuracy, all_f1_score))
        return

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
    est_midi_folder = '/home/neuralnet/segmentation_key/estimate_Roll/testing/'
    exp_dir = os.path.join(os.path.abspath('./experiments'), 'exp_5')
    with open(os.path.join(exp_dir,'hyperparams.json'), 'r') as hpfile:
        hp = json.load(hpfile)
    print(hp['best_loss'])
    print(hp['best_epoch'])
    checkpoints = 'checkpoint-{}.tar'.format(hp['best_epoch'])
    checkpoint = torch.load(os.path.join(exp_dir, checkpoints))
    test_loader = process_data()
    input_shape = (1, 51, 100)
    model = Generator(input_shape).cuda()
    model.load_state_dict(checkpoint['state_dict_G'])
    test(model, test_loader)