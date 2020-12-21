import os
import torch
import torch.optim as optim
import numpy as np
from torchvision.utils import save_image
import json
import torch.utils.data as utils
from Roll2MidiNet import Generator, Discriminator,weights_init_normal
from Roll2Midi_dataset import Roll2MidiDataset
from torch.autograd import Variable
from sklearn import metrics
torch.cuda.set_device(0)
cuda = torch.device("cuda")
print(torch.cuda.current_device())
Tensor = torch.cuda.FloatTensor

class hyperparams(object):
    def __init__(self):
        self.train_epoch = 300
        self.test_freq = 1
        self.exp_name = 'Correct_Roll2MidiNet'

        self.channels = 1
        self.h = 51 #input Piano key ranges
        self.w = 100 # 4 seconds, 100 frames predictions

        self.iter_train_g_loss = []
        self.iter_train_d_loss = []

        self.iter_test_g_loss = []
        self.iter_test_d_loss = []

        self.g_loss_history = []
        self.d_loss_history = []

        self.test_g_loss_history = []
        self.test_d_loss_history = []
        self.best_loss = 1e10
        self.best_epoch = 0

def process_data():
    train_dataset = Roll2MidiDataset(train=True)
    train_loader = utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = Roll2MidiDataset(train=False)
    test_loader = utils.DataLoader(test_dataset, batch_size=16)
    return train_loader, test_loader

def train(generator, discriminator, epoch, train_loader, optimizer_G, optimizer_D,
          scheduler, adversarial_loss, iter_train_g_loss, iter_train_d_loss):
    generator.train()
    discriminator.train()
    train_g_loss = 0
    train_d_loss = 0
    for batch_idx, data in enumerate(train_loader):
        gt, roll = data
        # Adversarial ground truths
        valid = Variable(Tensor(gt.shape[0], *discriminator.output_shape).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(gt.shape[0], *discriminator.output_shape).fill_(0.0), requires_grad=False)
        gt = gt.type(Tensor)
        roll = roll.type(Tensor)

        real = Variable(gt)
        roll_ = Variable(roll)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(roll_)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001*adversarial_loss(discriminator(gen_imgs), valid) + 0.999*adversarial_loss(gen_imgs, gt)

        g_loss.backward()

        iter_train_g_loss.append(g_loss.item())
        train_g_loss += g_loss

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()

        iter_train_d_loss.append(d_loss.item())
        train_d_loss += d_loss

        optimizer_D.step()

        if batch_idx % 2 == 0:
            print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\t g Loss: {4:.6f} | d Loss: {5:.6f}'.format(epoch, batch_idx * roll.shape[0],
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            g_loss.item() / roll.shape[0], d_loss.item() / roll.shape[0]))
    scheduler.step(train_g_loss / len(train_loader.dataset))
    print('====> Epoch: {} Average g loss: {:.4f} | d loss: {:.4f}'.format(epoch, train_g_loss / len(train_loader.dataset), train_d_loss / len(train_loader.dataset)))
    return train_g_loss / len(train_loader.dataset),train_d_loss / len(train_loader.dataset)

def test(generator, discriminator, epoch, test_loader, adversarial_loss,
         iter_test_g_loss,iter_test_d_loss):
    all_label = []
    all_pred_label = []
    all_pred_label_ = []
    with torch.no_grad():
        generator.eval()
        discriminator.eval()
        test_g_loss = 0
        test_d_loss = 0
        for idx, data in enumerate(test_loader):
            gt, roll = data
            # Adversarial ground truths
            valid = Variable(Tensor(gt.shape[0], *discriminator.output_shape).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(gt.shape[0], *discriminator.output_shape).fill_(0.0), requires_grad=False)
            gt = gt.type(Tensor)
            roll = roll.type(Tensor)

            real = Variable(gt)
            roll_ = Variable(roll)
            gen_imgs = generator(roll_)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(gen_imgs, gt)

            iter_test_g_loss.append(g_loss.item())
            test_g_loss += g_loss

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            iter_test_d_loss.append(d_loss.item())
            test_d_loss += d_loss

            pred_label = gen_imgs >= 0.4
            numpy_label = gt.cpu().detach().numpy().astype(np.int) # B,1,51, 50
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


        test_g_loss /= len(test_loader.dataset)
        test_d_loss /= len(test_loader.dataset)
        # scheduler.step(test_loss)
        print('====> Test set g loss: {:.4f} | d loss: {:.4f}'.format(test_g_loss, test_d_loss))

        all_label = np.vstack(all_label)
        all_pred_label = np.vstack(all_pred_label)
        all_precision = metrics.precision_score(all_label, all_pred_label, average='samples', zero_division=1)
        all_recall = metrics.recall_score(all_label, all_pred_label, average='samples', zero_division=1)
        all_f1_score = metrics.f1_score(all_label, all_pred_label, average='samples', zero_division=1)
        print(
            "Threshold 0.4, epoch {0}  avg precision:{1:.3f} | avg recall:{2:.3f} | f1 score:{3:.3f}".format(
                epoch, all_precision, all_recall, all_f1_score))

        all_pred_label_ = np.vstack(all_pred_label_)
        all_precision = metrics.precision_score(all_label, all_pred_label_, average='samples', zero_division=1)
        all_recall = metrics.recall_score(all_label, all_pred_label_, average='samples', zero_division=1)
        all_f1_score = metrics.f1_score(all_label, all_pred_label_, average='samples', zero_division=1)
        print(
            "Threshold 0.5, epoch {0}  avg precision:{1:.3f} | avg recall:{2:.3f} | f1 score:{3:.3f}".format(
                epoch, all_precision, all_recall, all_f1_score))


        return test_g_loss, test_d_loss


def main():
    hp = hyperparams()

    try:
        # the dir to save the Roll2Midi model
        exp_root = os.path.join(os.path.abspath('./'), 'Correct_Roll2Midi_experiments')
        os.makedirs(exp_root)
    except FileExistsError:
        pass

    exp_dir = os.path.join(exp_root, hp.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    input_shape = (hp.channels, hp.h, hp.w)
    # Loss function
    adversarial_loss = torch.nn.MSELoss()

    generator = Generator(input_shape)
    discriminator = Discriminator(input_shape)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    generator.cuda()
    discriminator.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.5*1e-3, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.5*1e-3, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', patience=2)
    train_loader, test_loader = process_data()
    print ('start training')
    for epoch in range(hp.train_epoch):
        # training loop
        g_loss, d_loss = train(generator, discriminator, epoch, train_loader, optimizer_G, optimizer_D,
                              scheduler, adversarial_loss, hp.iter_train_g_loss, hp.iter_train_d_loss)
        hp.g_loss_history.append(g_loss.item())
        hp.d_loss_history.append(d_loss.item())

        # test
        if epoch % hp.test_freq == 0:
            test_g_loss,test_d_loss = test(generator, discriminator,  epoch, test_loader, adversarial_loss,
                                           hp.iter_test_g_loss, hp.iter_test_d_loss)
            hp.test_g_loss_history.append(test_g_loss.item())
            hp.test_d_loss_history.append(test_d_loss.item())
            if test_g_loss + test_d_loss < hp.best_loss:
                torch.save({'epoch': epoch + 1, 'state_dict_G': generator.state_dict(),
                            'optimizer_G': optimizer_G.state_dict(),
                           'state_dict_D': discriminator.state_dict(),
                            'optimizer_D': optimizer_D.state_dict()},
                           os.path.join(exp_dir, 'checkpoint-{}.tar'.format(str(epoch + 1))))
                hp.best_loss = test_g_loss.item()+test_d_loss.item()
                hp.best_epoch = epoch + 1
                with open(os.path.join(exp_dir, 'hyperparams.json'), 'w') as outfile:
                    json.dump(hp.__dict__, outfile)

if __name__ == "__main__":
    main()