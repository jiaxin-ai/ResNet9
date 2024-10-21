import os

# from transform import *
from data.transform import *

class Dataset():
    def __init__(self, data_root, fraction=1.0):
        super(Dataset,self).__init__()
        self.data_root = data_root
        self.fraction = fraction
        self._initialize()

    def _initialize(self):
        self.img_path_list = []
        for img_name in os.listdir(self.data_root):
            if ('.jpg' not in img_name) and ('.png' not in img_name):
                continue
            self.img_path_list.append(os.path.join(self.data_root, img_name))
        self.img_path_list.sort(key=lambda x: int(x.split('_')[-2]))
        self.img_path_list = self.img_path_list[:int(len(self.img_path_list) * self.fraction)]
        # for img_path in self.img_path_list:
        #     print(img_path)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, item):
        img_path = self.img_path_list[item]
        img = normalize(img_path=img_path,mean = [0.1307], std = [0.3081])

        img_name = os.path.basename(img_path)
        label = np.array([int(img_name.split('.')[0].split('_')[2])], dtype=np.int64)

        return img, label

if __name__ == '__main__':
    mean = [0.1307]
    std = [0.3081]
    train_dataset = Dataset(data_root='../MNIST/images/train')
    # test_dataset = Dataset(data_root='../MNIST/images/test')
    print(len(train_dataset))
    # print(len(test_dataset))
    img1, label1 = train_dataset[0]
    print(img1.shape, label1)
    denormalize_and_save(img1,mean,std,'tmp2.jpg')
    # cv2.imwrite('tmp.jpg',img1)
    # print(train_dataset[0][0].shape. train_dataset[0][1])
    # print(test_dataset[0][0].shape. test_dataset[0][1])