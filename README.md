# ResNet and VGG implementation for CIFAR10/CIFAR100 in Pytorch
The purpose of this repo is to provide a valid pytorch implementation of ResNet-s and VGG-s for CIFAR10 and do comparative analysis.

The following models are provided (with validation results):
| Name      | # layers | # params| Test err(paper) | Test err(this impl.)|
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
