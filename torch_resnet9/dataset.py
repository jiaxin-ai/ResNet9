import os
from PIL import Image
import torchvision
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, img_transform=torchvision.transforms.ToTensor(), fraction=1.0):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.img_transform = img_transform
        self.fraction = fraction
        self._initialize()

    def _initialize(self):
        self.img_path_list = []
        for img_name in os.listdir(self.data_root):
            if ('.jpg' not in img_name) and ('.png' not in img_name):
                continue
            self.img_path_list.append(os.path.join(self.data_root,img_name))
        self.img_path_list.sort(key=lambda x:int(x.split('_')[-2]))
        self.img_path_list = self.img_path_list[:int(len(self.img_path_list)*self.fraction)]
        # for img_path in self.img_path_list:
        #     print(img_path)


    def __len__(self):
        return len(self.img_path_list)


    def __getitem__(self, item):
        img_path = self.img_path_list[item]

        img_name = os.path.basename(img_path)
        label = torch.tensor(int(img_name.split('.')[0].split('_')[2]))

        image = Image.open(img_path)
        if self.img_transform is not None:
            image = self.img_transform(image)

        return image, label

if __name__ == '__main__':
    pass
    # dataset = Dataset(data_root='./mnist/MNIST/images/test')
    # print(len(dataset))
    # img, label = dataset[0]
    # print(img.shape,label)
    # labels = label.unsqueeze(0)
    # print(labels.flatten())