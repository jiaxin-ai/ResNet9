import os

import time
import numpy as np
import cupy as cp

from models.model import Net
from data.dataset import Dataset
from data.dataloader import DataLoader
from loss.compute_crossEntropy import cross_entropy_loss_with_gradient
from optimizer.Adam import Adam
from util import get_gpu_info



class ResNet:
    def __init__(self, data_root, save_dir, batch_size, num_epochs=75, learning_rate=0.001, test_fr=1, verbose=True):
        self.data_root = data_root
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.start_epoch = 1
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.test_fr = test_fr
        self.verbose = verbose

        self.net = Net()

        self.train_dataset = Dataset(data_root=os.path.join(self.data_root,'train'), fraction=1)
        self.train_dataloader = DataLoader(data=self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                           drop_last=False)
        self.test_dataset = Dataset(data_root=os.path.join(self.data_root,'test'), fraction=1)
        self.test_dataloader = DataLoader(data=self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                          drop_last=False)

        print('Train Dataset and DataLoader length: ', len(self.train_dataset), len(self.train_dataloader))
        print('Test Dataset and DataLoader length: ', len(self.test_dataset), len(self.test_dataloader))

        self.optimizer = Adam(self.net.parameters(),learning_rate,weight_decay=1e-5)

        self.iter_infos = []
        self.epoch_infos = []



    def train(self,process):
        self.net.train()

        for epoch in range(self.start_epoch, 1+self.num_epochs):
            print(f'Epoch-{epoch} =========')

            epoch_correct = 0
            epoch_total = 0
            loss_total = []

            epoch_start_time = time.time()

            for i, data in enumerate(self.train_dataloader):
                iter_start_time = time.time()

                images, labels = data
                # print(f'{i}, {images.shape}, {labels.shape}, {labels[0,0]}')

                outputs = self.net.forward(images)
                loss, out_diff_tensor = cross_entropy_loss_with_gradient(outputs, labels)
                grads = self.net.backward(out_diff_tensor)
                self.optimizer.update(grads)

                # print('params: ',self.net.fc.params['weight'].flatten()[:3], self.net.fc.parameters()['bias'].flatten()[:3])

                loss_total.append(float(loss))
                predicted = cp.argmax(outputs, axis=1)
                batch_total = labels.shape[0]
                batch_correct = np.sum(predicted == labels.flatten())
                epoch_total += batch_total
                epoch_correct += batch_correct

                cpu_usage = process.cpu_percent(interval=None)  # 获取当前 CPU 占用率

                gpu_memory_usage, gpu_util = get_gpu_info()[0]['memory_used'], get_gpu_info()[0]['gpu_utilization']

                iter_info_str = f'Epoch: {epoch}, Step:{i}, Time: {time.time() - iter_start_time}, Loss: {loss}, Batch Accuracy: {batch_correct / batch_total}, Running Epoch Accuracy: {epoch_correct / epoch_total}, CPU usage: {cpu_usage}%, GPU memory usage: {gpu_memory_usage} MiB, GPU Util: {gpu_util}%'

                if (i+1) % self.test_fr == 0:
                    test_accuracy, test_info = self.test()
                    iter_info_str += f', {test_info}'

                self.iter_infos.append(iter_info_str)
                if self.verbose:
                    print(iter_info_str)
                # if i == 1000:
                #     exit(0)

            epoch_info_str = f'Epoch: {epoch}, Time: {time.time() - epoch_start_time}, Loss:{np.mean(loss_total)}, Train_Accuracy: {epoch_correct / epoch_total}'

            # if epoch % self.test_fr == 0:
            #     test_accuracy = self.test()
            #     epoch_info_str += f', Test Accuracy: {test_accuracy}'

            test_accuracy, test_info = self.test()
            epoch_info_str += f', {test_info}'

            self.epoch_infos.append(epoch_info_str)
            self.save_parameters(dir=self.save_dir, epoch=epoch)




    def test(self):
        self.net.eval()
        print('Testing......')
        test_start_time = time.time()

        correct = 0
        total = 0

        for i, data in enumerate(self.test_dataloader):
            images, labels = data

            outputs = self.net.forward(images)

            predicted = cp.argmax(outputs, axis=1)
            total = total + labels.shape[0]
            correct = correct + np.sum(predicted == labels.flatten())

        test_acc = correct / total
        test_info = f'Test time: {time.time() - test_start_time}, Test Acc: {test_acc}'
        print(test_info)
        self.net.train()
        return test_acc, test_info


    def save_parameters(self, dir, epoch):
        save_dir = os.path.join(dir,f'Epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        self.net.save(save_dir)

        with open(os.path.join(dir, 'iter_info.txt'), 'w') as f:
            for info in self.iter_infos:
                f.write(info+'\n')

        with open(os.path.join(dir, 'loss_and_acc_per_epoch.txt'), 'w') as f:
            for info in self.epoch_infos:
                f.write(info+'\n')


    def load_parameters(self, dir, epoch):
        self.net.load(os.path.join(dir,f'Epoch_{epoch}'))
        self.start_epoch = epoch




if __name__ == '__main__':
    resnet = ResNet(data_root = 'E:/PycharmProjects/mnist_resnet9_numpy/MNIST/images',
                    save_dir = './results',
                    batch_size = 256,
                    num_epochs = 1,
                    learning_rate = 0.001,
                    test_fr = 1,
                    verbose = True)
    resnet.train()