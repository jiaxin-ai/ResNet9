import cupy as cp

# Adam 优化器
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=1e-5):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.m = {}  # 存储一阶矩
        self.v = {}  # 存储二阶矩
        self.t = 0  # 全局时间步计数

    def update(self, grads):
        # print('in adam update')
        """在整个网络的backward结束后调用一次step方法"""
        self.t += 1

        # print(self.params)
        for param_name in self.params:
            param = self.params[param_name]
            grad = grads[param_name]

            # 添加权重衰减
            if 'weight' in param_name:
                grad += self.weight_decay * param

            if param_name not in self.m:
                # 初始化一阶和二阶矩
                self.m[param_name] = cp.zeros_like(grad)
                self.v[param_name] = cp.zeros_like(grad)

            # 更新一阶矩和二阶矩
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

            # 计算偏差修正
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

            # 更新参数
            # print('in adam', param_name,self.params[param_name], m_hat)
            self.params[param_name] -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)