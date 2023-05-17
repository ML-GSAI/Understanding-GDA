import torch

import torchvision
import torchvision.transforms as transforms

#this
DATA_DESC = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.48145466, 0.4578275, 0.40821073], 
    'std': [0.26862954, 0.26130258, 0.27577711],
}


def load_cifar10(data_dir, use_augmentation='base'):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)    
    return train_dataset, test_dataset