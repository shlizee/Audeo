from Video2Roll_dataset import Video2RollDataset
from torch.utils.data import DataLoader
import torch
from torch import optim

import Video2RollNet

from Video2Roll_solver import Solver
import torch.nn as nn
from balance_data import MultilabelBalancedRandomSampler

if __name__ == "__main__":
    train_dataset = Video2RollDataset(subset='train')
    train_sampler = MultilabelBalancedRandomSampler(train_dataset.train_labels)
    train_data_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    test_dataset = Video2RollDataset(subset='test')
    test_data_loader = DataLoader(test_dataset, batch_size=64)
    device = torch.device('cuda')

    net = Video2RollNet.resnet18()
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    solver = Solver(train_data_loader, test_data_loader, net, criterion, optimizer, scheduler, epochs=10)
    solver.train()
