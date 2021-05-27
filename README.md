## Self-Expressive Network

This repository contians the Pytorch implementation of our paper [Learning a Self-Expressive Networl for Subspace Clustering]()

### Usage

``` python
# dataset: MNIST, EMNIST, FashionMNIST, CIFAR10
python main.py --dataset=MNIST
```

### Feature Generation
The preprocessed features (Scattering transform for MNIST/FashonMNIST/EMNIST, MCR$^2$ for CIFAR10) are already provided in ./datasets folder. Code for feature generation for MNIST, Fashion and EMNIST are provided in feature generation.py. 
``` python
# dataset: MNIST, EMNIST, FashionMNIST, CIFAR10
python feature_generation.py --dataset=MNIST
```

For MCR$^2$ feature generation, please refer to [ryanchankh/mcr2](https://github.com/ryanchankh/mcr2).

### Contact
Please contact zhangshangzhi@bupt.edu.cn if you have any question on the codes.