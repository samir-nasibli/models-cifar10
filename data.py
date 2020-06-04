import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def train_loader(num_workers, batch_size, normalize):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    return loader


def validate_loader(num_workers, batch_size, normalize):
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return loader
