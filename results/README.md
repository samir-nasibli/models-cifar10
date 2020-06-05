## Experiment results
Some machine and env info:
```bash
Torch 1.5.0+cu101
Current device 0
Devices count: 1
Device: Tesla P100-PCIE-16GB
```
The provided table show comparative analysys of between ResNet and VGG models on Cifar10. Results are obtained for models, that trainded 300 epocs


| Name      | # layers | # params| Prec@1 |
|-----------|---------:|--------:|:-----------------:|
|[ResNet20](https://drive.google.com/file/d/11ASser28ZsYDNJPQzTHqEm5IL-mAaoJh/view?usp=sharing)   |    20    | 0.27M   |91.99%|
|[VGG11](https://drive.google.com/file/d/11niPBS9H8gmvF5JmR4l_ZGrPXz9Nr7s3/view?usp=sharing)   |    11    | 133M   |86.81%|
|[VGG16 (135 epc)](https://drive.google.com/file/d/1-iJdp3lIlgbWmoHAu-JcUgIlHVjUx4dU/view?usp=sharing)  |   16   |  138   | 91.86%|

## TODO
* Update Table
* Link to disk
* Results for `ResNet32`, `ResNet44`, `ResNet56`, `ResNet110`, `ResNet120`, `VGG11_bn`, `VGG13`, `VGG13_bn`, `VGG16_bn`, `VGG19`, `VGG19_bn` 
