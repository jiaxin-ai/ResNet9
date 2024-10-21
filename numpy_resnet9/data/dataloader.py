import numpy as np

class DataLoader:
    def __init__(self, data, batch_size, shuffle=True, drop_last=False):
        """
        用 NumPy 实现的简易 DataLoader
        :param data: 输入数据，假设 shape 为 (N, *)，N 是样本数量，* 表示其他维度
        :param batch_size: 每个批次的大小
        :param shuffle: 是否在每个 epoch 开始时打乱数据
        :param drop_last: 如果最后一个批次的数据量小于 batch_size，是否丢弃
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)  # 样本索引
        self.current_idx = 0  # 当前批次的起始位置

    def __iter__(self):
        """返回迭代器对象，并在每个 epoch 开始时打乱数据（如果需要）"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0  # 每个 epoch 从头开始
        return self

    def __next__(self):
        """加载下一个批次的数据"""
        if self.current_idx >= self.num_samples:
            raise StopIteration  # 当没有更多批次时，终止迭代

        # 计算当前批次的结束位置
        end_idx = self.current_idx + self.batch_size

        # 如果需要丢弃最后一个批次并且它不足 batch_size
        if self.drop_last and end_idx > self.num_samples:
            raise StopIteration

        # 提取当前批次的索引
        batch_indices = self.indices[self.current_idx:end_idx]

        # 提取数据
        # batch_data = self._extract_batch(batch_indices)
        batch_data = [[] for _ in range(len(self.data[0]))]  # 初始化列表以存储每种数据
        for idx in batch_indices:
            for i, value in enumerate(self.data[idx]):  # 遍历每个元组的元素
                batch_data[i].append(value)  # 收集每个元素的数据

        for i in range(len(batch_data)):
            batch_data[i] = np.stack(batch_data[i], axis=0)
        # batch_data = self.data[batch_indices]

        # 更新 current_idx 为下一个批次的起始位置
        self.current_idx = end_idx

        return batch_data

    def _extract_batch(self, batch_indices):
        """
        根据索引提取批次数据，避免动态堆叠操作，直接利用 NumPy 索引。
        """
        # 假设 self.data 是 (img, label, ...) 形式的元组数据
        first_sample = self.data[batch_indices[0]]

        # 确定批次中每个数据类型的 shape，并预分配内存
        batch = [np.zeros((len(batch_indices), *np.shape(element))) for element in first_sample]

        # 填充批次数据
        for i, idx in enumerate(batch_indices):
            for j, element in enumerate(self.data[idx]):
                batch[j][i] = element

        return batch

    def __len__(self):
        """返回每个 epoch 中的批次数量"""
        if self.drop_last:
            return self.num_samples // self.batch_size  # 向下取整
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size  # 向上取整

if __name__ == '__main__':
    from dataset import Dataset

    train_dataset = Dataset(data_root='../MNIST/images/train')
    train_dataloader = DataLoader(data=train_dataset, batch_size=256, shuffle=True, drop_last=True)
    test_dataset = Dataset(data_root='../MNIST/images/test')
    test_dataloader = DataLoader(data=test_dataset,batch_size=256,shuffle=False,drop_last=False)

    print(len(train_dataloader))
    print(len(test_dataloader))

    # for i, batch in enumerate(train_dataloader,0):
    #     print(i)
    #     print(batch[0][1])
    #     break
    # 使用 DataLoader 迭代数据
    for epoch in range(1):  # 两个 epoch
        print(f"Epoch {epoch + 1}:")
        for i, batch in enumerate(train_dataloader):
            img, label = batch
            print(label)
            # print(img.shape, label.shape, label)
            break
        print("-" * 20)

    # # 使用 DataLoader 迭代数据
    # for epoch in range(1):  # 两个 epoch
    #     print(f"Test Epoch {epoch + 1}:")
    #     for batch in test_dataloader:
    #         img, label = batch
    #         print(img.shape, label.shape)
    #         break
    #     print("-" * 20)