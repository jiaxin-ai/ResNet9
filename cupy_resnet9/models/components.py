import os
import cupy as cp


class conv_layer:

    def __init__(self, in_channels, out_channels, kernel_h, kernel_w, padding=1, stride=1, bias=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.params = {}
        self.init_param()

    def init_param(self):
        self.kernel = cp.random.uniform(
            low=-cp.sqrt(6.0 / (self.out_channels + self.in_channels * self.kernel_h * self.kernel_w)),
            high=cp.sqrt(6.0 / (self.in_channels + self.out_channels * self.kernel_h * self.kernel_w)),
            size=(self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)
        ).astype(cp.float32)
        self.params['weight'] = self.kernel

        if self.bias:
            self.bias = cp.zeros([self.out_channels], dtype=cp.float32)
            self.params['bias'] = self.bias
        else:
            self.bias = None

    @staticmethod
    def pad(in_tensor,pad_h,pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        padded = cp.zeros([batch_num, in_channels, in_h + 2 * pad_h, in_w + 2 * pad_w])
        padded[:, :, pad_h:pad_h + in_h, pad_w:pad_w + in_w] = in_tensor
        return padded

    @staticmethod
    def convolution(in_tensor, kernel, stride=1, dilate=1):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        out_channels = kernel.shape[0]
        assert kernel.shape[1] == in_channels

        kernel_h = kernel.shape[2]
        kernel_w = kernel.shape[3]

        out_h = int((in_h - kernel_h + 1) / stride)
        out_w = int((in_w - kernel_w + 1) / stride)

        kernel = kernel.reshape(out_channels, -1)

        extend_in = cp.zeros([in_channels * kernel_h * kernel_w, batch_num * out_h * out_w])
        for i in range(out_h):
            for j in range(out_w):
                part_in = in_tensor[:, :, i * stride:i * stride + kernel_h, j * stride:j * stride + kernel_w].reshape(
                    batch_num, -1)
                extend_in[:, (i * out_w + j) * batch_num:(i * out_w + j + 1) * batch_num] = part_in.T

        out_tensor = cp.dot(kernel, extend_in)
        out_tensor = out_tensor.reshape(out_channels, out_h * out_w, batch_num)
        out_tensor = out_tensor.transpose(2, 0, 1).reshape(batch_num, out_channels, out_h, out_w)

        return out_tensor

    def forward(self, in_tensor):
        in_tensor = self.pad(in_tensor,self.padding,self.padding)
        self.in_tensor = in_tensor.copy()

        self.out_tensor = conv_layer.convolution(in_tensor, self.kernel, self.stride)

        if self.bias:
            self.out_tensor += self.bias.reshape(1, self.out_channels, 1, 1)

        return self.out_tensor


    def backward(self, out_diff_tensor):
        assert out_diff_tensor.shape == self.out_tensor.shape

        batch_num = out_diff_tensor.shape[0]
        out_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]

        extend_out = cp.zeros([batch_num, out_channels, out_h, out_w, self.stride * self.stride])
        extend_out[:, :, :, :, 0] = out_diff_tensor
        extend_out = extend_out.reshape(batch_num, out_channels, out_h, out_w, self.stride, self.stride)
        extend_out = extend_out.transpose(0, 1, 2, 4, 3, 5).reshape(batch_num, out_channels, out_h * self.stride,
                                                                    out_w * self.stride)

        kernel_diff = conv_layer.convolution(self.in_tensor.transpose(1, 0, 2, 3), extend_out.transpose(1, 0, 2, 3))
        kernel_diff = kernel_diff.transpose(1, 0, 2, 3)

        padded = conv_layer.pad(extend_out, self.kernel_h - 1, self.kernel_w - 1)
        kernel_trans = self.kernel.reshape(self.out_channels, self.in_channels, self.kernel_h * self.kernel_w)
        kernel_trans = kernel_trans[:, :, ::-1].reshape(self.kernel.shape)
        self.in_diff_tensor = conv_layer.convolution(padded, kernel_trans.transpose(1, 0, 2, 3))
        assert self.in_diff_tensor.shape == self.in_tensor.shape

        self.in_diff_tensor = self.in_diff_tensor[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # self.kernel -= lr * kernel_diff

        if self.bias:
            bias_diff = cp.sum(out_diff_tensor, axis=(0, 2, 3)).reshape(self.bias.shape)
            # self.bias -= lr * bias_diff
            self.grads = {'weight': kernel_diff, 'bias': bias_diff}
        else:
            self.grads = {'weight': kernel_diff}


    def parameters(self):
        return self.params

    def get_grads(self):
        return self.grads


    def save(self, path, conv_num):
        if os.path.exists(path) == False:
            os.mkdir(path)

        cp.save(os.path.join(path, "conv{}_weight.npy".format(conv_num)), self.kernel)
        if self.bias:
            cp.save(os.path.join(path, "conv{}_bias.npy".format(conv_num)), self.bias)

        return conv_num + 1


    def load(self, path, conv_num):
        assert os.path.exists(path)

        self.kernel = cp.load(os.path.join(path, "conv{}_weight.npy".format(conv_num)))
        if self.bias:
            self.bias = cp.load(os.path.join(path, "conv{}_bias.npy").format(conv_num))

        return conv_num + 1

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


class bn_layer:
    def __init__(self, num_features, momentum = 0.9):
        self.gamma = cp.random.uniform(low=0, high=1, size=num_features).astype(cp.float32)
        self.bias = cp.zeros([num_features], dtype=cp.float32)
        self.moving_avg = cp.zeros([num_features])
        self.moving_var = cp.ones([num_features])
        self.num_features = num_features
        self.momentum = momentum
        self.is_train = True
        self.epsilon = 1e-5

        self.params = {'weight': self.gamma, 'bias': self.bias}

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def forward(self, in_tensor):
        assert in_tensor.shape[1] == self.num_features

        self.in_tensor = in_tensor.copy()

        if self.is_train:
            mean = in_tensor.mean(axis=(0,2,3))
            var = in_tensor.var(axis=(0,2,3))
            self.moving_avg = mean * (1-self.momentum) + self.momentum * self.moving_avg
            self.moving_var = var * (1-self.momentum) + self.momentum * self.moving_var
            self.var = var
            self.mean = mean
        else:
            mean = self.moving_avg
            var = self.moving_var

        self.normalized = (in_tensor - mean.reshape(1,-1,1,1)) / cp.sqrt(var.reshape(1, -1, 1, 1) + self.epsilon)
        out_tensor = self.gamma.reshape(1,-1,1,1) * self.normalized + self.bias.reshape(1,-1,1,1)

        return out_tensor

    def backward(self, out_diff_tensor):
        assert out_diff_tensor.shape == self.in_tensor.shape
        assert self.is_train

        m = self.in_tensor.shape[0] * self.in_tensor.shape[2] * self.in_tensor.shape[3]

        normalized_diff = self.gamma.reshape(1,-1,1,1) * out_diff_tensor
        var_diff = -0.5 * cp.sum(normalized_diff * self.normalized, axis=(0, 2, 3)) / (self.var + self.epsilon)
        mean_diff = -1.0 * cp.sum(normalized_diff, axis=(0, 2, 3)) / cp.sqrt(self.var + self.epsilon)
        in_diff_tensor1 = normalized_diff / cp.sqrt(self.var.reshape(1, -1, 1, 1) + self.epsilon)
        in_diff_tensor2 = var_diff.reshape(1,-1,1,1) * (self.in_tensor - self.mean.reshape(1,-1,1,1)) * 2 / m
        in_diff_tensor3 = mean_diff.reshape(1,-1,1,1) / m
        self.in_diff_tensor = in_diff_tensor1 + in_diff_tensor2 + in_diff_tensor3

        gamma_diff = cp.sum(self.normalized * out_diff_tensor, axis=(0, 2, 3))
        # self.gamma -= lr * gamma_diff

        bias_diff = cp.sum(out_diff_tensor, axis=(0, 2, 3))
        # self.bias -= lr * bias_diff

        self.grads = {'weight': gamma_diff, 'bias': bias_diff}


    def parameters(self):
        return self.params

    def get_grads(self):
        return self.grads


    def save(self, path, bn_num):
        if os.path.exists(path) == False:
            os.mkdir(path)

        cp.save(os.path.join(path, "bn{}_weight.npy".format(bn_num)), self.gamma)
        cp.save(os.path.join(path, "bn{}_bias.npy".format(bn_num)), self.bias)
        cp.save(os.path.join(path, "bn{}_mean.npy".format(bn_num)), self.moving_avg)
        cp.save(os.path.join(path, "bn{}_var.npy".format(bn_num)), self.moving_var)

        return bn_num + 1

    def load(self, path, bn_num):
        assert os.path.exists(path)

        self.gamma = cp.load(os.path.join(path, "bn{}_weight.npy".format(bn_num)))
        self.bias = cp.load(os.path.join(path, "bn{}_bias.npy".format(bn_num)))
        self.moving_avg = cp.load(os.path.join(path, "bn{}_mean.npy".format(bn_num)))
        self.moving_var = cp.load(os.path.join(path, "bn{}_var.npy".format(bn_num)))

        return bn_num + 1


class relu:

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, in_tensor):
        self.in_tensor = in_tensor.copy()
        self.out_tensor = in_tensor.copy()
        self.out_tensor[self.in_tensor < 0.0] = 0.0
        return self.out_tensor

    def backward(self, out_diff_tensor):
        assert self.out_tensor.shape == out_diff_tensor.shape
        self.in_diff_tensor = out_diff_tensor.copy()
        self.in_diff_tensor[self.in_tensor < 0.0] = 0.0

    def parameters(self):
        return self.params

    def get_grads(self):
        return self.grads

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


class max_pooling:
    def __init__(self, kernel_h, kernel_w, stride):
        assert stride > 1
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride

        self.params = {}
        self.grads = {}

    @staticmethod
    def pad(in_tensor, pad_h, pad_w):
        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        padded = cp.zeros([batch_num, in_channels, in_h + 2 * pad_h, in_w + 2 * pad_w])
        padded[:, :, pad_h:pad_h + in_h, pad_w:pad_w + in_w] = in_tensor
        return padded

    def forward(self, in_tensor):
        self.shape = in_tensor.shape

        batch_num = in_tensor.shape[0]
        in_channels = in_tensor.shape[1]
        in_h = in_tensor.shape[2]
        in_w = in_tensor.shape[3]
        out_h = int((in_h - self.kernel_h) / self.stride) + 1
        out_w = int((in_w - self.kernel_w) / self.stride) + 1

        out_tensor = cp.zeros([batch_num, in_channels, out_h, out_w])
        self.maxindex = cp.zeros([batch_num, in_channels, out_h, out_w], dtype=cp.int32)
        for i in range(out_h):
            for j in range(out_w):
                part = in_tensor[:, :, i * self.stride:i * self.stride + self.kernel_h,
                       j * self.stride:j * self.stride + self.kernel_w].reshape(batch_num, in_channels, -1)
                out_tensor[:, :, i, j] = cp.max(part, axis=-1)
                self.maxindex[:, :, i, j] = cp.argmax(part, axis=-1)
        self.out_tensor = out_tensor
        return self.out_tensor

    def backward(self, out_diff_tensor):
        assert out_diff_tensor.shape == self.out_tensor.shape
        batch_num = out_diff_tensor.shape[0]
        in_channels = out_diff_tensor.shape[1]
        out_h = out_diff_tensor.shape[2]
        out_w = out_diff_tensor.shape[3]
        in_h = self.shape[2]
        in_w = self.shape[3]

        out_diff_tensor = out_diff_tensor.reshape(batch_num * in_channels, out_h, out_w)
        self.maxindex = self.maxindex.reshape(batch_num * in_channels, out_h, out_w)

        self.in_diff_tensor = cp.zeros([batch_num * in_channels, in_h, in_w])
        h_index = (self.maxindex / self.kernel_h).astype(cp.int32)
        w_index = self.maxindex - h_index * self.kernel_h
        for i in range(out_h):
            for j in range(out_w):
                self.in_diff_tensor[
                    range(batch_num * in_channels), i * self.stride + h_index[:, i, j], j * self.stride + w_index[:, i,
                                                                                                          j]] += out_diff_tensor[
                                                                                                                 :, i,
                                                                                                                 j]
        self.in_diff_tensor = self.in_diff_tensor.reshape(batch_num, in_channels, in_h, in_w)


    def parameters(self):
        return self.params

    def get_grads(self):
        return self.grads

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False



class FC:   # without sigmoid

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.init_param()

    def init_param(self):   # Xavier初始化
        self.kernel = cp.random.uniform(
            low = -cp.sqrt(6.0 / (self.out_features + self.in_features)),
            high = cp.sqrt(6.0 / (self.in_features + self.out_features)),
            size = (self.out_features, self.in_features)
        ).astype(cp.float32)
        # self.kernel = np.zeros([self.out_features, self.in_features]) # 对拍用0初始化
        self.bias = cp.zeros([self.out_features], dtype=cp.float32)
        self.params = {'weight': self.kernel, 'bias': self.bias}

    def forward(self, in_tensor):
        self.shape = in_tensor.shape
        self.in_tensor = in_tensor.reshape(in_tensor.shape[0], -1).copy()
        assert self.in_tensor.shape[1] == self.kernel.shape[1]
        self.out_tensor = cp.dot(self.in_tensor, self.kernel.T) + self.bias.T

        # # sigmoid
        # self.out_tensor = 1.0 / (1.0 + np.exp(-self.out_tensor))

        return self.out_tensor


    # 没有sigmoid版backward
    def backward(self, out_diff_tensor):    # out_diff_tensor：损失函数对y的导数
        assert out_diff_tensor.shape == self.out_tensor.shape
        # kernel_diff = np.dot(out_diff_tensor.T, self.in_tensor).squeeze()
        kernel_diff = cp.dot(self.in_tensor.T, out_diff_tensor).T
        # bias_diff = np.sum(out_diff_tensor, axis=0).reshape(self.bias.shape)
        bias_diff = cp.mean(out_diff_tensor, axis=0).reshape(self.bias.shape)
        self.in_diff_tensor = cp.dot(out_diff_tensor, self.kernel).reshape(self.shape)
        self.grads = {'weight': kernel_diff, 'bias': bias_diff}
        
    # def backward(self, out_diff_tensor):    # out_diff_tensor：损失函数对y的倒数
    #     assert out_diff_tensor.shape == self.out_tensor.shape
    #     nonlinear_diff = self.out_tensor * (1 - self.out_tensor) * out_diff_tensor
    #     kernel_diff = np.dot(nonlinear_diff.T, self.in_tensor).squeeze()
    #     bias_diff = np.sum(nonlinear_diff, axis=0).reshape(self.bias.shape)
    #     self.in_diff_tensor = np.dot(nonlinear_diff, self.kernel).reshape(self.shape)
    #     self.grads = {'weight': kernel_diff, 'bias': bias_diff}

    def parameters(self):
        return self.params

    def get_grads(self):
        return self.grads

    def save(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)

        cp.save(os.path.join(path, "fc_weight.npy"), self.kernel)
        cp.save(os.path.join(path, "fc_bias.npy"), self.bias)

    def load(self, path):
        assert os.path.exists(path)

        self.kernel = cp.load(os.path.join(path, "fc_weight.npy"))
        self.bias = cp.load(os.path.join(path, "fc_bias.npy"))


    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False