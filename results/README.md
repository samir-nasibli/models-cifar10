## Experiment results
Some machine and env info:
```bash
Torch 1.5.0+cu101
Current device 0
Devices count: 1
Device: Tesla P100-PCIE-16GB
```
The provided table show comparative analysys of ResNet20 on Cifar10 between this impl. and original [paper](https://arxiv.org/abs/1512.03385)


| Name      | # layers | # params| Test err(this impl.) | Test err(papers)|
|-----------|---------:|--------:|:-----------------:|:---------------------:|
|[ResNet20]()   |    20    | 0.27M   |8.91%| 8.27%|
|[VGG11]()   |    -    | -   |TODO| -|
|[VGG16]()  |   -    |  -   | TODO| -|

## TODO
* Update Table
* Link to disk
* Results for `ResNet32`, `ResNet44`, `ResNet56`, `ResNet110`, `ResNet120`, `VGG11_bn`, `VGG13`, `VGG13_bn`, `VGG16_bn`, `VGG19`, `VGG19_bn` 
