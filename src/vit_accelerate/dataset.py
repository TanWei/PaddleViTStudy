from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.vision import datasets
from paddle.vision import transforms
import math

#对于ImageNet2012Dataset（ILSVRC2012）的加载查看：class ImageNet2012Dataset(Dataset):

def get_transforms(mode='train'):
#     if mode == 'train':
#         transforms_train = transforms.Compose([
#         transforms.RandomResizedCrop(size=(224, 224),
#                                      interpolation='bicubic'),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD)])
#     else:
# s       scale_size = int(math.floor(config.DATA.IMAGE_SIZE / config.DATA.CROP_PCT))
#         transforms_val = transforms.Compose([
#         transforms.Resize(scale_size, 'bicubic'), # single int for resize shorter side of image
#         transforms.CenterCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD)])
    if mode == 'train':
        data_transforms = transforms.Compose([
           transforms.RandomResizedCrop(size=(224, 224),interpolation='bicubic'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]) 
    else:
        scale_size = int(math.floor(224 / 0.875))
        data_transforms = transforms.Compose([
            transforms.Resize(scale_size, 'bicubic'), # single int for resize shorter side of image
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    return data_transforms


def get_dataset(name='cifar10', mode='train'):
    if name == 'cifar10':
        dataset = datasets.Cifar10(data_file='./cifar-10-python.tar.gz',mode=mode, transform=get_transforms(mode))

    return dataset

def get_dataloader(dataset, batch_size=128, mode='train'):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=(mode == 'train'))
    return dataloader
