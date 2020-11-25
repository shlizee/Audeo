import time
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import _classification

class Solver(object):

    def __init__(self, data_loader, test_data_loader, model, criterion, optimizer, lr_scheduler, epochs):
        self.save_model_path = './models/Video2Roll.pth' # change to your path
        self.test_loader = test_data_loader
        self.data_loader = data_loader
        self.net = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        # Training config
        self.epochs = epochs
        # logging
        self.step = 0
        self.global_step = 0
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.val_loss = torch.zeros(self.epochs)
        self.visdom = True
        self.visdom_epoch = 1
        self.visdom_id = 'key classification'
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'val loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)

    def train(self):
        # Train model multi-epoches
        pre_val_loss = 1e4
        for epoch in range(self.epochs):
            print("Training...")
            self.net.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            # training loop
            tr_avg_loss, tr_avg_precision, tr_avg_recall = self.train_loop()

            # evaluate
            self.net.eval()
            val_avg_loss, val_avg_precision, val_avg_recall, val_avg_acc, val_fscore = self.validate()
            print('-' * 85)
            print('Train Summary | Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                epoch+1, time.time() - start, tr_avg_loss, tr_avg_precision, tr_avg_recall))
            print("epoch {0} validation loss:{1:.3f} | avg precision:{2:.3f} | avg recall:{3:.3f} | avg acc:{4:.3f} | f1 score:{5:.3f}".format(
                epoch+1, val_avg_loss, val_avg_precision, val_avg_recall, val_avg_acc, val_fscore))
            print('-' * 85)

            if val_avg_loss < pre_val_loss:
                pre_val_loss = val_avg_loss
                torch.save(self.net.state_dict(), self.save_model_path)
            # Save model each epoch
            self.val_loss[epoch] = val_avg_loss
            self.tr_loss[epoch] = tr_avg_loss

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                # train_y_axis = self.tr_loss[0:epoch+1]
                # val_x_axis = self.vis_epochs[0:epoch+1:10]
                # val_y_axis = self.val_loss[0:epoch//10+1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.val_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def train_loop(self):
        data_loader = self.data_loader
        epoch_loss = 0
        epoch_precision = 0
        epoch_recall = 0
        count = 0
        start = time.time()

        for i, data in enumerate(data_loader):
            imgs, label = data
            logits = self.net(imgs)
            loss = self.criterion(logits,label)
            # set the threshold of the logits
            pred_label = torch.sigmoid(logits) >= 0.4
            numpy_label = label.cpu().detach().numpy().astype(np.int)
            numpy_pre_label = pred_label.cpu().detach().numpy().astype(np.int)

            precision = metrics.precision_score(numpy_label,numpy_pre_label, average='samples', zero_division=1)
            recall = metrics.recall_score(numpy_label,numpy_pre_label, average='samples', zero_division=1)
            if self.global_step % 100 == 0:
                end = time.time()
                print(
                    "step {0} loss:{1:.4f} | precision:{2:.3f} | recall:{3:.3f} | time:{4:.2f}".format(self.global_step, loss.item(), precision,
                                                                                        recall,end - start))
                start = end

            epoch_precision += precision
            epoch_recall += recall
            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            count += 1
            self.global_step += 1
        self.lr_scheduler.step(epoch_loss / count)
        return epoch_loss/count, epoch_precision/count, epoch_recall/count

    def validate(self):
        epoch_loss = 0
        count = 0
        all_pred_label = []
        all_label = []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                imgs, label = data
                logits = self.net(imgs)
                loss = self.criterion(logits, label)
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

