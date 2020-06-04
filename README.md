# prototype-cifar10
prototype for cifar10 pet project

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
