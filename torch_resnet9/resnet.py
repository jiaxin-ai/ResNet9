import time

import torch
import torchvision.transforms as transforms
import os
import numpy as np

import model
from dataset import Dataset
from util import get_gpu_info


class ResNet:

    def __init__(self, data_root):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.net = model.Net().cuda() if self.use_cuda else model.Net()
        self.optimizer = None

        self.train_accuracies = []
        self.test_accuracies = []
        self.start_epoch = 1

        self.iter_infos = []
        self.epoch_infos = []

        self.data_root = data_root

    def train(self, process, save_dir, num_epochs=75, batch_size=256, learning_rate=0.001, test_fr=500, verbose=True):
        """Trains the network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_each_epoch : boolean
            True: Test the network after every training epoch, False: no testing
        verbose : boolean
            True: Print training progress to console, False: silent mode
        """
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.net.train()

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), # mnist
        ])

        # train_transform = transforms.Compose([    # cifar dataset
        #     util.Cutout(num_cutouts=2, size=8, p=0.8),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        # train_dataset = datasets.CIFAR10('data/cifar', train=True, download=True, transform=train_transform)
        train_dataset = Dataset(data_root=os.path.join(self.data_root,'train'),img_transform=train_transform)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()


        for epoch in range(self.start_epoch, 1 + num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            epoch_correct = 0
            epoch_total = 0

            loss_total = []
            epoch_start_time = time.time()

            for i, data in enumerate(data_loader, 1):
                iter_start_time = time.time()
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                loss = criterion(outputs, labels.squeeze_())

                loss_total.append(loss.item())

                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, dim=1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels.flatten()).sum().item()

                epoch_total += batch_total
                epoch_correct += batch_correct

                cpu_usage = process.cpu_percent(interval=None)  # 获取当前 CPU 占用率

                gpu_memory_usage, gpu_util = get_gpu_info()[0]['memory_used'], get_gpu_info()[0]['gpu_utilization']

                iter_info_str = f'Epoch: {epoch}, Step:{i}, Time: {time.time() - iter_start_time}, Loss: {loss}, Batch Accuracy: {batch_correct / batch_total}, Running Epoch Accuracy: {epoch_correct / epoch_total}, CPU usage: {cpu_usage}%, GPU memory usage: {gpu_memory_usage} MiB, GPU Util: {gpu_util}%'

                if (i+1) % test_fr == 0:
                    test_accuracy, test_info = self.test()
                    iter_info_str += f', {test_info}'
                    self.test_accuracies.append(test_accuracy)

                self.iter_infos.append(iter_info_str)
                if verbose:
                    print(iter_info_str)

            epoch_info_str = f'Epoch: {epoch}, Time: {time.time() - epoch_start_time}, Loss:{np.mean(loss_total)}, Train_Accuracy: {epoch_correct / epoch_total}'
            self.train_accuracies.append(epoch_correct / epoch_total)

            # if test_each_epoch:
            #     test_accuracy = self.test()
            #     epoch_info_str += f', Test Accuracy: {test_accuracy}'
            #     self.test_accuracies.append(test_accuracy)
            #     # if verbose:
            #     #     print('Test Acc: {}'.format(test_accuracy))

            test_accuracy, test_info = self.test()
            epoch_info_str += f', {test_info}'
            self.test_accuracies.append(test_accuracy)

            # Save parameters after every epoch
            self.epoch_infos.append(epoch_info_str)
            self.save_parameters(epoch, directory=save_dir)

    def test(self, batch_size=256):
        """Tests the network.

        """
        print('Testing......')
        test_start_time = time.time()
        self.net.eval()

        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,)),
                                             ])

        # test_dataset = datasets.CIFAR10('data/cifar', train=False, download=True, transform=test_transform)
        test_dataset = Dataset(data_root=os.path.join(self.data_root,'test'),img_transform=test_transform)
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs, dim=1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels.flatten()).sum().item()
        test_acc = correct / total
        test_info = f'Test time: {time.time() - test_start_time}, Test Acc: {test_acc}'
        print(test_info)
        self.net.train()
        return test_acc, test_info

    def save_parameters(self, epoch, directory):
        """Saves the parameters of the network to the specified directory.

        Parameters
        ----------
        epoch : int
            The current epoch
        directory : str
            The directory to which the parameters will be saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
        }, os.path.join(directory, 'resnet_' + str(epoch) + '.pth'))

        with open(os.path.join(directory, 'iter_info.txt'), 'w') as f:
            for info in self.iter_infos:
                f.write(info + '\n')

        with open(os.path.join(directory, 'loss_and_acc_per_epoch.txt'), 'w') as f:
            for info in self.epoch_infos:
                f.write(info + '\n')

    def load_parameters(self, path):
        """Loads the given set of parameters.

        Parameters
        ----------
        path : str
            The file path pointing to the file containing the parameters
        """
        self.optimizer = torch.optim.Adam(self.net.parameters())
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        self.start_epoch = checkpoint['epoch']



