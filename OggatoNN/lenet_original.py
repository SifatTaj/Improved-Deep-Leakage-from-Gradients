import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse

from matplotlib import pyplot as plt
from torchvision.io import read_image
from aug_utils.aug_conv_2d import AnonConv2d

from utils import progress_bar

# from ptflops import get_model_complexity_info
from datetime import datetime

start_time = time.perf_counter()


class AnonLeNet(nn.Module):
    def __init__(self, num_classes=10, aug_indices=None, deanon_dim=None, rand_aug_indices=None):
        super(AnonLeNet, self).__init__()

        self.lenet = nn.Sequential(
            AnonConv2d(1, 6, kernel_size=5, stride=1, padding=0, aug_indices=aug_indices, deanon_dim=28),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        self.fake_net = nn.Sequential(
            AnonConv2d(1, 6, kernel_size=5, stride=1, padding=0, aug_indices=rand_aug_indices, deanon_dim=28),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # out = self.anon_conv(x)
        out = self.lenet(x)
        # out_fake = self.fake_conv(x)
        out_fake = self.fake_net(x)
        return out, out_fake


# class MyLeNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(MyLeNet, self).__init__()
#
#         self.anon_conv = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
#         self.lenet = nn.Sequential(
#             nn.BatchNorm2d(6),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten(1),
#             nn.Linear(256, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, 10)
#         )
#     def forward(self, x):
#         out = self.anon_conv(x)
#         out = self.lenet(out)
#         return out

class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

        act_fake = nn.Sigmoid
        self.body_fake = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act_fake(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act_fake(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act_fake(),
        )
        self.fc_fake = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out_fake = self.body_fake(x)
        out = out.view(out.size(0), -1)
        out_fake = out_fake.view(out.size(0), -1)
        out = self.fc(out)
        out_fake = self.fc(out_fake)
        return out


class AugDataset2(torch.utils.data.Dataset):
    def __init__(self, aug_dataset, original_dataset):
        # self.x = torch.from_numpy(np.array([data.numpy().astype(np.float16) for data in aug_dataset[:, 0]]))
        self.x = aug_dataset
        self.x = self.x.float()

        print("set:", self.x.shape)

        # self.y = torch.from_numpy(original_dataset[:, 1].astype('int'))
        self.y = torch.IntTensor([data[1] for data in original_dataset])
        self.y = self.y.long()

        self.n_samples = aug_dataset.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train(net, epoch, trainloader, device, optimizer, criterion):
    train_losses = []
    train_accuracy = []

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        all_outputs = net(inputs)
        all_outputs = all_outputs if type(all_outputs) is tuple else [all_outputs]
        for idx, outputs in enumerate(all_outputs):
            optimizer[idx].zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer[idx].step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_losses.append(train_loss)
    train_accuracy.append(100. * correct / total)

    return train_losses, train_accuracy


def test(net, epoch, testloader, device, criterion):
    test_losses = []
    test_accuracy = []

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            all_outputs = net(inputs)
            all_outputs = all_outputs if type(all_outputs) is tuple else [all_outputs]
            for outputs in all_outputs:
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_losses.append(test_loss)
    test_accuracy.append(100. * correct / total)

    return test_losses, test_accuracy


def oggato_trainer():
    ds_name = 'MNIST'
    aug_percent = 0
    num_channel = 1
    device = 'cuda'

    original_trainset = torchvision.datasets.MNIST(root='data_torch',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

    original_testset = torchvision.datasets.MNIST(root='data_torch',
                                                  train=False,
                                                  transform=transforms.ToTensor(),
                                                  download=True)

    original_testloader = torch.utils.data.DataLoader(
        original_testset, batch_size=128, shuffle=False, num_workers=2)

    original_trainloader = torch.utils.data.DataLoader(
        original_trainset, batch_size=128, shuffle=False, num_workers=2)

    if aug_percent > 0:
        aug_trainset = torch.load(f'aug_datasets/{ds_name}_train_{aug_percent}.pt')
        aug_trainset = list(aug_trainset.parameters())[0]
        aug_trainset = AugDataset2(aug_trainset, original_trainset)
        aug_trainloader = torch.utils.data.DataLoader(aug_trainset, batch_size=128, shuffle=True, num_workers=2)

        aug_testset = torch.load(f'aug_datasets/{ds_name}_test_{aug_percent}.pt')
        aug_testset = list(aug_testset.parameters())[0]
        aug_testset = AugDataset2(aug_testset, original_testset)
        aug_testloader = torch.utils.data.DataLoader(aug_testset, batch_size=128, shuffle=True, num_workers=2)

        aug_indices_all = torch.load(f'aug_datasets/{ds_name}_indices.pt')

        aug_indices = []
        for c in range(num_channel):
            aug_index = aug_indices_all[4 - 1][c]
            aug_index = aug_index.numpy().astype(int)
            aug_index = aug_index[aug_index != 0]
            aug_indices.append(aug_index)

        aug_indices = np.array(aug_indices)
        rand_aug_indices = []
        rand_aug_indices.append(random.sample(range(0, 56 * 56), 2352))

    shape_img = (28, 28)
    num_classes = 10
    channel = 1
    hidden = 588

    net = LeNet(num_classes=num_classes, channel=channel, hideen=hidden) if aug_percent == 0 else AnonLeNet(aug_indices=aug_indices, deanon_dim=28,
                                                     rand_aug_indices=rand_aug_indices)
    # net = MyLeNet()
    # net = AnonResNet18(10, num_channel, aug_indices, 32, aug_percent)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizers = []
    optimizers.append(optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4))
    # optimizers.append(optim.SGD(net.fake_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4))

    # optimizer = optim.SGD(net.lenet.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, 1 + 10):
        if aug_percent > 0:
            train_losses, train_accuracy = train(net, epoch, aug_trainloader, device=device, optimizer=optimizers,
                                                 criterion=criterion)
            test_losses, test_accuracy = test(net, epoch, aug_testloader, device=device, criterion=criterion)
            # scheduler.step()
        else:
            train_losses, train_accuracy = train(net, epoch, original_trainloader, device=device, optimizer=optimizers,
                                                 criterion=criterion)
            test_losses, test_accuracy = test(net, epoch, original_testloader, device=device, criterion=criterion)
            # scheduler.step()

    end_time = time.perf_counter() - start_time
    print('Total time for OggatoNN', end_time)

    # net = net.to('cpu')

    print(test_accuracy)
    return net

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

if __name__ == '__main__':
    # net = oggato_trainer()
    num_classes = 10
    channel = 1
    hidden = 588

    net = LeNet(num_classes=num_classes, channel=channel, hideen=hidden)
    net.apply(weights_init)
    net.to('cuda')

    dst = torchvision.datasets.MNIST(root='data_torch', download=True)

    lr = 1.0
    num_dummy = 1
    Iteration = 100
    num_exp = 10
    num_classes = 10

    dataset = 'MNIST'
    root_path = '.'
    save_path = os.path.join(root_path, 'results/iDLG_%s' % dataset).replace('\\', '/')

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    device = 'cuda'
    net = net.to(device)
    idx_shuffle = np.random.permutation(len(dst))

    for method in ['DLG', 'iDLG']:
        print('%s, Try to generate %d images' % (method, num_dummy))

        criterion = nn.CrossEntropyLoss().to(device)
        imidx_list = []

        for imidx in range(num_dummy):
            idx = idx_shuffle[imidx]
            imidx_list.append(idx)
            tmp_datum = tt(dst[idx][0]).float().to(device)
            tmp_datum = tmp_datum.view(1, *tmp_datum.size())
            tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
            tmp_label = tmp_label.view(1, )
            if imidx == 0:
                gt_data = tmp_datum
                gt_label = tmp_label
            else:
                gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                gt_label = torch.cat((gt_label, tmp_label), dim=0)

        # compute original gradient
        out = net(gt_data)
        y = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        # generate dummy data and label
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

        if method == 'DLG':
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
        elif method == 'iDLG':
            optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
            # predict the ground-truth label
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
                (1,)).requires_grad_(False)

        history = []
        history_iters = []
        losses = []
        mses = []
        train_iters = []

        print('lr =', lr)
        for iters in range(Iteration):

            def closure():
                optimizer.zero_grad()
                pred = net(dummy_data)
                if method == 'DLG':
                    dummy_loss = - torch.mean(
                        torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    # dummy_loss = criterion(pred, gt_label)
                elif method == 'iDLG':
                    dummy_loss = criterion(pred, label_pred)

                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff


            optimizer.step(closure)
            current_loss = closure().item()
            train_iters.append(iters)
            losses.append(current_loss)
            mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

            if iters % int(Iteration / 30) == 0:
                current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                history_iters.append(iters)

                for imidx in range(num_dummy):
                    plt.figure(figsize=(12, 8))
                    plt.subplot(3, 10, 1)
                    plt.imshow(tp(gt_data[imidx].cpu()))
                    for i in range(min(len(history), 29)):
                        plt.subplot(3, 10, i + 2)
                        plt.imshow(history[i][imidx])
                        plt.title('iter=%d' % (history_iters[i]))
                        plt.axis('off')
                    if method == 'DLG':
                        plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                        plt.close()
                    elif method == 'iDLG':
                        plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                        plt.close()

                if current_loss < 0.000001:  # converge
                    break

        if method == 'DLG':
            loss_DLG = losses
            label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
            mse_DLG = mses
        elif method == 'iDLG':
            loss_iDLG = losses
            label_iDLG = label_pred.item()
            mse_iDLG = mses

    print('imidx_list:', imidx_list)
    print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
    print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
    print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)

    print('----------------------\n\n')
