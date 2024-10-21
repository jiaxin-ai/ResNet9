from resnet import ResNet

resnet = ResNet(
    data_root = 'E:/PycharmProjects/mnist_resnet9_numpy/MNIST/images',
    save_dir = './results/exp_final',
    batch_size = 256,
    num_epochs = 2,
    learning_rate = 0.001,
    test_fr = 50,
    verbose = True
)
resnet.load_parameters(dir='results/exp_final',epoch=2)
resnet.test()