import numpy as np

import torch
import pickle
import argparse

from decomposition.dim_reduction import dim_reduction
from kymatio import Scattering2D
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MNIST")
args = parser.parse_args()

if not args.dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
    raise Exception("Only MNIST, FashionMNIST and EMNIST are currently supported.")

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

scattering = Scattering2D(J=3, shape=(32, 32))
if use_cuda:
    scattering = scattering.cuda()

transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])
train_idx = []
test_idx = []
print("Loading dataset...")
if args.dataset == "EMNIST":
    split = 'byclass'
    EMNIST_train = datasets.EMNIST('./datasets', train=True, download=False, transform=transform, split=split)
    EMNIST_test = datasets.EMNIST('./datasets', train=False, download=False, transform=transform, split=split)
    EMNIST_train_loader = torch.utils.data.DataLoader(EMNIST_train, batch_size=len(EMNIST_train), shuffle=False)
    EMNIST_test_loader = torch.utils.data.DataLoader(EMNIST_test, batch_size=len(EMNIST_test), shuffle=False)

    raw_train_data, label_train = next(iter(EMNIST_train_loader))  
    raw_test_data, label_test = next(iter(EMNIST_test_loader))
    target_classes = list(range(36, 62)) # class id for lowercase letters
    train_idx = list(filter(lambda x: label_train[x] in target_classes, range(len(EMNIST_train))))
    test_idx = list(filter(lambda x: label_test[x] in target_classes, range(len(EMNIST_test))))
elif args.dataset == "MNIST":
    MNIST_train = datasets.MNIST('./datasets', train=True, download=False, transform=transform)
    MNIST_test = datasets.MNIST('./datasets', train=False, download=False, transform=transform)
    MNIST_train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=len(MNIST_train), shuffle=False)
    MNIST_test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)
    raw_train_data, label_train = next(iter(MNIST_train_loader))  
    raw_test_data, label_test = next(iter(MNIST_test_loader))
    train_idx = list(range(len(MNIST_train)))
    test_idx = list(range(len(MNIST_test)))
elif args.dataset == "FashionMNIST":
    FashionMNIST_train = datasets.FashionMNIST('./datasets', train=True, download=False, transform=transform)
    FashionMNIST_test = datasets.FashionMNIST('./datasets', train=False, download=False, transform=transform)
    FashionMNIST_train_loader = torch.utils.data.DataLoader(FashionMNIST_train, batch_size=len(FashionMNIST_train), shuffle=False)
    FashionMNIST_test_loader = torch.utils.data.DataLoader(FashionMNIST_test, batch_size=len(FashionMNIST_test), shuffle=False)
    raw_train_data, label_train = next(iter(FashionMNIST_train_loader))  
    raw_test_data, label_test = next(iter(FashionMNIST_test_loader))
    train_idx = list(range(len(FashionMNIST_train)))
    test_idx = list(range(len(FashionMNIST_test)))

print("Dataset Size: ", len(train_idx) + len(test_idx)) # Should be 190998 according to ESC paper
        
raw_train_data, label_train = raw_train_data[train_idx], label_train[train_idx]
raw_test_data, label_test = raw_test_data[test_idx], label_test[test_idx]

label = torch.cat((label_train, label_test), 0)

print('Computing scattering on {}...'.format(args.dataset))

use_cuda = False

if use_cuda:
    raw_train_data = raw_train_data.cuda()
    raw_test_data = raw_test_data.cuda()

train_data = scattering(raw_train_data) 
test_data = scattering(raw_test_data)
data = torch.cat((train_data, test_data), 0)

print('Data preprocessing....')
n_sample = data.shape[0]

# scattering transform normalization
data = data.cpu().numpy().reshape(n_sample, data.shape[2], -1)
image_norm = np.linalg.norm(data, ord=np.inf, axis=2, keepdims=True)  # infinity norm of each transform
data = data / image_norm  # normalize each scattering transform to the range [-1, 1]
data = data.reshape(n_sample, -1)  # fatten and concatenate all transforms

# dimension reduction
data = dim_reduction(data, 500)  # dimension reduction by PCA

label = label.numpy()          
train_size = label_train.shape[0]
test_size = label_test.shape[0]

with open('./{}_scattering_train_data.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(data[:train_size], f)
with open('./{}_scattering_test_data.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(data[train_size:], f)
with open('./{}_scattering_train_label.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(label[:train_size], f)
with open('./{}_scattering_test_label.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(label[train_size:], f)
print("Done.")