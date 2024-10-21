import cupy as cp

def cross_entropy_loss_with_gradient(logits, labels):
    # 1. 计算 softmax 输出，减去 logits 的最大值进行数值稳定处理
    logits_exp = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
    softmax_out = logits_exp / cp.sum(logits_exp, axis=1, keepdims=True)

    # 2. 计算交叉熵损失
    N = logits.shape[0]
    p_label = softmax_out[cp.arange(N), labels.squeeze(-1)]

    # 避免 log(0) 的情况，加一个小的 epsilon
    epsilon = 1e-12
    loss = -cp.log(p_label + epsilon)  # 负对数损失
    # loss_mean = np.sum(loss) / N # 平均损失
    loss_mean = cp.mean(loss)  # 平均损失

    # 3. 计算梯度
    # 初始化梯度为 softmax 输出
    gradient = softmax_out.copy()
    # 将真实标签对应的位置减去 1
    gradient[cp.arange(N), labels.squeeze(-1)] -= 1
    # 计算平均梯度
    gradient = gradient / N

    return loss_mean, gradient


if __name__ == '__main__':
    # 示例 logits
    x = cp.array([[1.2, 3.4, 0.3, 5.3, 0.4],
                  [1.6, 0.4, 2.3, 4.3, 3.4]])

    # 真实标签 (一维向量)
    y = cp.array([[1], [3]])

    # NumPy 实现
    loss, grad_numpy = cross_entropy_loss_with_gradient(x, y)
    print("NumPy Loss:", loss)
    print("NumPy Gradient:\n", grad_numpy)

    # PyTorch 实现
    import torch
    x_torch = torch.tensor([[1.2, 3.4, 0.3, 5.3, 0.4],
                            [1.6, 0.4, 2.3, 4.3, 3.4]], requires_grad=True)
    y_torch = torch.tensor([[1], [3]])  # 一维向量
    print("x_torch shape:", x_torch.shape)
    print("y_torch shape:", y_torch.shape)

    # 使用 PyTorch 的 CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()
    loss_torch = criterion(x_torch, y_torch.squeeze_())
    loss_torch.backward()

    print("PyTorch Loss:", loss_torch.item())
    print("PyTorch Gradient:\n", x_torch.grad)
