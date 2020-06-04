# ResNet and VGG implementation for CIFAR10/CIFAR100 in Pytorch
The purpose of this repo is to provide a valid pytorch implementation of ResNet-s and VGG-s for CIFAR10 and do comparative analysis.
<<<<<<< HEAD

The following models are provided (with validation results):
| Name      | # layers | # params| Test err(this impl.) | Test err(papers)|
|-----------|---------:|--------:|:-----------------:|:---------------------:|
|[ResNet20]()   |    20    | 0.27M   |TODO| -|
|[ResNet32]()  |    32    | 0.46M   | TODO| -|
|[ResNet44]()   |    44    | 0.66M   | TODO| -|
|[ResNet56]()   |    56    | 0.85M   | TODO| -|
|[ResNet110]()  |   110    |  1.7M   | TODO| -|
|[ResNet1202]() |  1202    | 19.4M   | TODO| -|
|[VGG11]()   |    -    | -   |TODO| -|
|[VGG11_bn]()  |    -    | -   | TODO| -|
|[VGG13]()   |    -    | -   | TODO| -|
|[VGG13_bn]()   |    -    | -   | TODO| -|
|[VGG16]()  |   -    |  -   | TODO| -|
|[VGG16_bn]() |  -    | -   | TODO| -|
|[VGG19]()  |   -    |  -   | TODO| -|
|[VGG169_bn]() |  -    | -   | TODO| -|
## Prerequisites
* Python3.5+
* CUDA 10.1

## Installation
Python setuptools and python package manager (pip) install packages into system directory by default.  The training code tested only via [virtual environment](https://docs.python.org/3/tutorial/venv.html).

In order to use virtual environment you should install it first:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv -p `which python3` <env_dir>
```

Before starting to work inside virtual environment, it should be activated:

```bash
source <env_dir>/bin/activate
```

Install dependencies using:

```
pip install -r requirements.txt
```

## How to run?
See the usage of runner:
```bash
python main.py -h
```
```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--half] [--cpu]
               [--save-dir SAVE_DIR]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: resnet110 | resnet1202 | resnet20
                        | resnet32 | resnet44 | resnet56 | vgg11 | vgg11_bn |
                        vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn
                        (default: vgg19)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 128)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 5e-4)
  --print-freq N, -p N  print frequency (default: 20)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --half                use half-precision(16-bit)
  --cpu                 use cpu
  --save-dir SAVE_DIR   The directory used to save the trained models
  --save-every SAVE_EVERY
                        Saves checkpoints at every specified number of epochs
```

Example of running:
```bash
python main.py  --arch=resnet20 --epochs=100  --save-dir=save_resnet20
```
For evaluation:
```bash
python main.py --evaluate --arch=resnet20  --save-dir=save_resnet20
```
