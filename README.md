# ResNet-9 implementation using Pytorch, Numpy, and CuPy, respectively.
This is an experiment homework implementing **ResNet-9** ([He et al. 2015](https://arxiv.org/pdf/1512.03385)) training on **MNIST** dataset, using **Numpy**, **CuPy** and **Pytorch**, respectively.
  * The PyTorch implementation uses the PyTorch framework to train the model easily.
  * The NumPy implementation manually handles parameter initialization for each layer, forward propagation, backpropagation, cross-entropy loss, and gradient calculation, along with a simplified dataloader and Adam optimizer. 
  * The CuPy implementation modifies the NumPy version to run on the GPU.

### This is a great project for beginners to get started with deep learning!

## Dependencies
* python 3.12
* torch 2.4.1+cu121
* torchvision 0.19.1+cu121
* numpy 1.26.3
* pillow

## Dataset
MNIST dataset can be downloaded using `torchvision.dataset`, or through the link [[Google Driver] MNIST](https://drive.google.com/file/d/1hPF78vS-o94fe-L5n1fALaUxxT7QEG8L/view?usp=drive_link).

## Training
### Preparation
#### Installation
```
conda create -n resnet python=3.12
conda activate resnet
pip install numpy==1.26.3
pip install pillow
pip install torchvision==0.19.1 torch==2.4.2 --index-url https://download.pytorch.org/whl/cu121
```
#### Note
* The Numpy implementation runs on CPU.
* The Pytorch and CuPy implementations both run on GPU.
  * The CUDA version is required to be `12.1`.
  * We expect users to have GPU with at least 8000M memory.

### Train
The hyper-parameters can be modified in `train.py `. The results will be saved in `results/exp_final`.
```
python train.py
```

### Results

<img src="https://github.com/jiaxin-ai/ResNet9/blob/main/pics/compare_loss_curves.png" width="750px">


## Inference with the pretrained models
### Pretrained models
* Create a new folder `results/exp_final` in the three directories: `torch_resnet9`, `numpy_resnet9`, `cupy_resnet9`. Take `torch_resnet9` as an example.
```
cd ./torch_resnet9
mkdir results
cd ./results
mkdir exp_final
```
* Download and copy the corresponding files or folders from the link to the respective `exp_final` folder.
[[Google Drive]](https://drive.google.com/drive/folders/1XdnKON7DEN-e2psaSaHFIEDVUyaNLTUo?usp=drive_link)

### Inference
```
python test.py
```
### Results

|  Epoch  |  Implementation  |  Train Accuracy (%)  |   Inference time (s)   |    Test Accuracy (%)   |
|:-----:|:-----:|:-----:|:-----:|:-----:|
|  1  |  Numpy  |  92.73  |  337.59  |  98.56  |
|    |  CuPy  |  92.61  |  93.95  |  98.15  |
|    |  Pytorch  |  94.80  |  4.39  |  98.49  |
|  2  |  Numpy  |  98.73  |  338.34  |  98.22  |
|    |  CuPy  |  98.80  |  61.24  |  98.96  |
|    |  Pytorch  |  98.97  |  4.36  |  99.04  |


## Acknowledgements
* [cifar10-resnet](https://github.com/matthias-wright/cifar10-resnet)
* [Numpy-Implementation-of-ResNet](https://github.com/lyzustc/Numpy-Implementation-of-ResNet)
