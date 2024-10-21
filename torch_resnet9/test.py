from resnet import ResNet


net = ResNet(data_root='E:/PycharmProjects/mnist_resnet9_numpy/MNIST/images')
net.load_parameters('results/exp_final/resnet_2.pth')
net.test(batch_size=256)
