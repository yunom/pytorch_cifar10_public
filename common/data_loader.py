import os
import sys
from collections import Counter
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from common.utils import *
from common import define


def load_data(name, batch_size, num_workers, val_split_ratio=0, pretrained=False, show_stats=False):
    if name == 'cifar10':
        train_set, val_set, test_set = load_cifar10(val_split_ratio, pretrained)
    else:
        raise RuntimeError('Invalid dataset name.')

    # create dataloader
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders['validate'] = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    output_dataset_info(dataloaders, show_stats)
    return dataloaders


def load_cifar10(val_split_ratio=0, pretrained=False):
    if pretrained:
        # when using pretrained model, use ImageNet statistics
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        # when not using pretrained model, use original Cifar-10 statistics
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    train_set = torchvision.datasets.CIFAR10(
        root=define.DATA_DIR, train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(
        root=define.DATA_DIR, train=True, download=True, transform=val_test_transform)
    test_set = torchvision.datasets.CIFAR10(
        root=define.DATA_DIR, train=False, download=True, transform=val_test_transform)
    sys.stdout = old_stdout

    # split train_set to train_set and val_set
    if val_split_ratio > 0:
        train_indices, val_indices = train_test_split(
            list(range(len(train_set))), test_size=val_split_ratio, stratify=train_set.targets)
        train_set = torch.utils.data.Subset(train_set, train_indices)
        val_set = torch.utils.data.Subset(val_set, val_indices)
    else:
        val_set = test_set

    return train_set, val_set, test_set


def output_dataset_info(dataloaders, show_stats=False):
    print('# show dataset information')
    if 'train' in dataloaders:
        dataset = dataloaders["train"].dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        print(f'  num of classes: {len(dataset.classes)}')
        print(f'  class labels  : {dataset.class_to_idx}')
    for data_label, dataloader in dataloaders.items():
        print(f'-- {data_label} dataset')
        dataset = dataloader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            full_dataset = dataset.dataset
            # get data length of subset
            dataset_count = len(dataset)
            # get data shape of subset
            dataset_shape = list(full_dataset.data.shape)
            dataset_shape[0] = dataset_count
            dataset_shape = tuple(dataset_shape)
            # count subset count per class
            subset_classes = [full_dataset.targets[i] for i in dataset.indices]
            count_per_class = Counter(subset_classes)
        else:
            dataset_shape = dataset.data.shape
            dataset_count = len(dataset)
            count_per_class = Counter(dataset.targets)
        print(f'  data shape : {dataset_shape}')  # (b, H, W, c)
        print(f'  num of data: {dataset_count}')
        print(f'  num of each class: {sorted(count_per_class.items(), key=lambda x: x[0])}')
        if show_stats:
            mean, std = get_mean_and_std_sequence(dataset)
            print(f'  mean : {mean}')
            print(f'  std  : {std}')


def output_trainset_stats(trainset):
    print('# show train dataset statistics')
    print('-- before transform')
    mean, std = get_mean_and_std_bulk(trainset)
    print(f'  mean : {mean}')
    print(f'  std  : {std}')
    print('-- after transform')
    mean, std = get_mean_and_std_sequence(trainset)
    print(f'  mean : {mean}')
    print(f'  std  : {std}')


def get_classes(dataloader):
    dataset = dataloader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset.classes


if __name__ == '__main__':
    load_data('cifar10', 128, 2, 0.2, show_stats=True)
