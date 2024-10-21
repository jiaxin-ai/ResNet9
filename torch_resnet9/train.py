import psutil
from resnet import ResNet

process = psutil.Process()

net = ResNet(data_root='E:/PycharmProjects/mnist_resnet9_numpy/MNIST/images')
net.train(
    process=process,
    save_dir='results/exp_final',
    num_epochs=2,
    batch_size=256,
    learning_rate=0.001,
    test_fr=50,
    verbose=True
)
