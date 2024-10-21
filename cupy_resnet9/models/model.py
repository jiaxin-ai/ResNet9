import cupy as cp
from models.components import *


class ResidualBlock():
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_h=kernel_size, kernel_w=kernel_size, padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = bn_layer(num_features=out_channels, momentum=0.9)
        self.conv_res2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_h=kernel_size, kernel_w=kernel_size, padding=padding, bias=False)
        self.conv_res2_bn = bn_layer(num_features=out_channels, momentum=0.9)

        self.params = {}

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = [
                conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_h=1, kernel_w=1, stride=stride, bias=False),
                bn_layer(num_features=out_channels, momentum=0.9)
            ]
            self.params['conv_res1_weight'] = self.conv_res1.parameters()['weight']
            self.params['conv_res1_bn_weight'] = self.conv_res1_bn.parameters()['weight']
            self.params['conv_res1_bn_bias'] = self.conv_res1_bn.parameters()['bias']
            self.params['conv_res2_weight'] = self.conv_res2.parameters()['weight']
            self.params['conv_res2_bn_weight'] = self.conv_res2_bn.parameters()['weight']
            self.params['conv_res2_bn_bias'] = self.conv_res2_bn.parameters()['bias']
            self.params['downsample_conv_weight'] = self.downsample[0].parameters()['weight']
            self.params['downsample_bn_weight'] = self.downsample[1].parameters()['weight']
            self.params['downsample_bn_bias'] = self.downsample[1].parameters()['bias']
        else:
            self.downsample = None
            self.params['conv_res1_weight'] = self.conv_res1.parameters()['weight']
            self.params['conv_res1_bn_weight'] = self.conv_res1_bn.parameters()['weight']
            self.params['conv_res1_bn_bias'] = self.conv_res1_bn.parameters()['bias']
            self.params['conv_res2_weight'] = self.conv_res2.parameters()['weight']
            self.params['conv_res2_bn_weight'] = self.conv_res2_bn.parameters()['weight']
            self.params['conv_res2_bn_bias'] = self.conv_res2_bn.parameters()['bias']

        self.relu1 = relu()
        self.relu2 = relu() # 2个relu，不然梯度不更新，相当于inplace=False


    def parameters(self):
        return self.params


    def train(self):
        self.conv_res1.train()
        self.conv_res1_bn.train()
        self.conv_res2.train()
        self.conv_res2_bn.train()
        if self.downsample is not None:
            for layer in self.downsample:
                layer.train()

    def eval(self):
        self.conv_res1.eval()
        self.conv_res1_bn.eval()
        self.conv_res2.eval()
        self.conv_res2_bn.eval()
        if self.downsample is not None:
            for layer in self.downsample:
                layer.eval()

    def forward(self, in_tensor):
        x = in_tensor.copy()
        residual = in_tensor.copy()   # residual

        out = self.relu1.forward(self.conv_res1_bn.forward(self.conv_res1.forward(x)))
        out = self.conv_res2_bn.forward(self.conv_res2.forward(out))

        if self.downsample is not None:
            for path in self.downsample:
                residual = path.forward(residual)

        # out = self.relu(out)  # official里这里有问题
        out = out + residual
        self.out = self.relu2.forward(out)
        return self.out


    def backward(self, out_diff_tensor):
        assert self.out.shape == out_diff_tensor.shape

        self.grads = {}

        self.relu2.backward(out_diff_tensor)
        x1 = self.relu2.in_diff_tensor
        x2 = x1.copy()

        self.conv_res2_bn.backward(x1)
        self.grads['conv_res2_bn_weight'] = self.conv_res2_bn.get_grads()['weight']
        self.grads['conv_res2_bn_bias'] = self.conv_res2_bn.get_grads()['bias']
        x1 = self.conv_res2_bn.in_diff_tensor

        self.conv_res2.backward(x1)
        self.grads['conv_res2_weight'] = self.conv_res2.get_grads()['weight']
        x1 = self.conv_res2.in_diff_tensor

        self.relu1.backward(x1)
        x1 = self.relu1.in_diff_tensor

        self.conv_res1_bn.backward(x1)
        self.grads['conv_res1_bn_weight'] = self.conv_res1_bn.get_grads()['weight']
        self.grads['conv_res1_bn_bias'] = self.conv_res1_bn.get_grads()['bias']
        x1 = self.conv_res1_bn.in_diff_tensor

        self.conv_res1.backward(x1)
        self.grads['conv_res1_weight'] = self.conv_res1.get_grads()['weight']
        x1 = self.conv_res1.in_diff_tensor

        if self.downsample is not None:
            self.downsample[1].backward(x2)
            self.grads['downsample_bn_weight'] = self.downsample[1].get_grads()['weight']
            self.grads['downsample_bn_bias'] = self.downsample[1].get_grads()['bias']
            x2 = self.downsample[1].in_diff_tensor

            self.downsample[0].backward(x2)
            self.grads['downsample_conv_weight'] = self.downsample[0].get_grads()['weight']
            x2 = self.downsample[0].in_diff_tensor

        self.in_diff_tensor = x1 + x2


    def get_grads(self):
        return self.grads

    def save(self, path, conv_num, bn_num):
        conv_num = self.conv_res1.save(path, conv_num)
        bn_num = self.conv_res1_bn.save(path, bn_num)
        conv_num = self.conv_res2.save(path, conv_num)
        bn_num = self.conv_res2_bn.save(path, bn_num)

        if self.downsample is not None:
            conv_num = self.downsample[0].save(path, conv_num)
            bn_num = self.downsample[1].save(path, bn_num)

        return conv_num, bn_num


    def load(self, path, conv_num, bn_num):
        conv_num = self.conv_res1.load(path, conv_num)
        bn_num = self.conv_res1_bn.load(path, bn_num)
        conv_num = self.conv_res2.load(path, conv_num)
        bn_num = self.conv_res2_bn.load(path, bn_num)

        if self.downsample is not None:
            conv_num = self.downsample[0].load(path, conv_num)
            bn_num = self.downsample[1].load(path, bn_num)

        return conv_num, bn_num



class Net:
    """
    A Residual network.
    """
    def __init__(self):
        super(Net, self).__init__()

        self.conv = [
            conv_layer(in_channels=1, out_channels=64, kernel_h=3, kernel_w=3, stride=1, padding=3, bias=False),
            bn_layer(num_features=64, momentum=0.9),    # train
            relu(),
            conv_layer(in_channels=64, out_channels=128, kernel_h=3, kernel_w=3, stride=1, padding=1, bias=False),
            bn_layer(num_features=128, momentum=0.9),   # train
            relu(),
            max_pooling(kernel_h=2, kernel_w=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),   # train
            conv_layer(in_channels=128, out_channels=256, kernel_h=3, kernel_w=3, stride=1, padding=1, bias=False),
            bn_layer(num_features=256, momentum=0.9),   # train
            relu(),
            max_pooling(kernel_h=2, kernel_w=2, stride=2),
            conv_layer(in_channels=256, out_channels=256, kernel_h=3, kernel_w=3, stride=1, padding=1, bias=False),
            bn_layer(num_features=256, momentum=0.9),   # train
            relu(),
            max_pooling(kernel_h=2, kernel_w=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),   # train
            max_pooling(kernel_h=2, kernel_w=2, stride=2)
        ]

        # self.fc = FC(in_features=784, out_features=10)
        self.fc = FC(in_features=1024, out_features=10)

    def train(self):
        for layer in self.conv:
            layer.train()
        self.fc.train()

    def eval(self):
        for layer in self.conv:
            layer.eval()
        self.fc.eval()

    def forward(self, in_tensor):
        x = in_tensor
        for layer in self.conv:
            x = layer.forward(x)
        out = self.fc.forward(x)
        return out


    def backward(self, out_diff_tensor):
        x = out_diff_tensor

        grads = {}

        self.fc.backward(x)
        fc_grads = self.fc.get_grads()
        for name in fc_grads:
            grads[f'{name}_{id(self.fc)}'] = fc_grads[name]
        x = self.fc.in_diff_tensor

        for i in range(1, len(self.conv) + 1):
            layer = self.conv[-i]
            layer.backward(x)
            layer_grads = layer.get_grads()
            for name in layer_grads:
                grads[f"{name}_{id(layer)}"] = layer_grads[name]
            x = layer.in_diff_tensor

        self.in_diff_tensor = x
        return grads


    def parameters(self):
        params = {}
        for layer in self.conv:
            layer_params = layer.parameters()
            for name, param in layer_params.items():
                params[f"{name}_{id(layer)}"] = param
        fc_params = self.fc.parameters()
        for name in fc_params:
            params[f'{name}_{id(self.fc)}'] = fc_params[name]
        return params


    def inference(self, in_tensor):
        out_tensor = self.forward(in_tensor).reshape(in_tensor.shape[0], -1)
        return cp.argmax(out_tensor, axis=1)


    def save(self, path):
        conv_num = 0
        bn_num = 0

        os.makedirs(path, exist_ok=True)

        conv_num = self.conv[0].save(path, conv_num)
        bn_num = self.conv[1].save(path, bn_num)
        conv_num = self.conv[3].save(path, conv_num)
        bn_num = self.conv[4].save(path, bn_num)
        conv_num,bn_num = self.conv[7].save(path=path,conv_num=conv_num,bn_num=bn_num)
        conv_num = self.conv[8].save(path, conv_num)
        bn_num = self.conv[9].save(path, bn_num)
        conv_num = self.conv[12].save(path, conv_num)
        bn_num = self.conv[13].save(path, bn_num)
        conv_num, bn_num = self.conv[16].save(path,conv_num, bn_num)
        self.fc.save(path)

    def load(self, path):
        conv_num = 0
        bn_num = 0

        conv_num = self.conv[0].load(path, conv_num)
        bn_num = self.conv[1].load(path, bn_num)
        conv_num = self.conv[3].load(path, conv_num)
        bn_num = self.conv[4].load(path, bn_num)
        conv_num, bn_num = self.conv[7].load(path, conv_num, bn_num)
        conv_num = self.conv[8].load(path, conv_num)
        bn_num = self.conv[9].load(path, bn_num)
        conv_num = self.conv[12].load(path, conv_num)
        bn_num = self.conv[13].load(path, bn_num)
        conv_num, bn_num = self.conv[16].load(path, conv_num, bn_num)
        self.fc.load(path)