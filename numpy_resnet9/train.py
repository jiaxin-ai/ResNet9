import psutil
from resnet import ResNet

process = psutil.Process()

resnet = ResNet(data_root = 'E:/PycharmProjects/mnist_resnet9_numpy/MNIST/images',
                    save_dir = './results/exp_final',
                    batch_size = 256,
                    num_epochs = 2,
                    learning_rate = 0.001,
                    test_fr = 50,
                    verbose = True)
resnet.train(process)