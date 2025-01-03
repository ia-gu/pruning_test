import os
import json

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

def build_dataset(args):
    train_transform, _ = build_transform(args)

    if args.dataset == 'CIFAR100':
        data_path = '~/dataset'
        dataset = datasets.CIFAR100(data_path, train=True, transform=train_transform, download=True)
        nb_classes = 100
    elif args.dataset == 'CIFAR10':
        data_path = '~/dataset'
        dataset = datasets.CIFAR10(data_path, train=True, transform=train_transform, download=True)
        nb_classes = 10
    elif args.dataset == 'ImageNet':
        data_path = '/data01/imagenet/train_256'
        dataset = datasets.ImageFolder(data_path, transform=train_transform)
        nb_classes = 1000
    elif args.dataset == 'TINY':
        data_path = '~/dataset/Tiny-ImageNet/tiny-imagenet-200/train'
        dataset = datasets.ImageFolder(data_path, transform=train_transform)
        nb_classes = 200
    print(args.dataset, nb_classes)

    return dataset, nb_classes

def build_eval_dataset(args):
    _, test_transform = build_transform(args)

    if args.dataset == 'CIFAR100':
        data_path = '~/dataset'
        dataset = datasets.CIFAR100(data_path, train=False, transform=test_transform, download=True)
        nb_classes = 100
    elif args.dataset == 'CIFAR10':
        data_path = '~/dataset'
        dataset = datasets.CIFAR10(data_path, train=False, transform=test_transform, download=True)
        nb_classes = 10
    elif args.dataset == 'ImageNet':
        data_path = '/data01/imagenet/val_256'
        dataset = datasets.ImageFolder(data_path, transform=test_transform)
        nb_classes = 1000
    elif args.dataset == 'TINY':
        data_path = '~/dataset/Tiny-ImageNet/tiny-imagenet-200/val'
        dataset = datasets.ImageFolder(data_path, transform=test_transform)
        nb_classes = 200
    print(args.dataset, nb_classes)

    return dataset, nb_classes

def build_test_dataset(args):
    _, test_transform = build_transform(args)

    if args.dataset == 'CIFAR10':
        dataset_normal = datasets.CIFAR10(data_path, train=False, transform=test_transform, download=True)
        root = os.path.join(data_path, 'CIFAR-10-C-ImageFolder')
        print(root)
        
    elif args.dataset == 'CIFAR100':
        dataset_normal = datasets.CIFAR100(data_path, train=False, transform=test_transform, download=True)
        root = os.path.join(data_path, 'CIFAR-100-C-ImageFolder')
        print(root)
    
    elif args.dataset == 'ImageNet':
        root = '/data01/imagenet/val_256'
        dataset_normal = datasets.ImageFolder(data_path, transform=test_transform)
        root = '/data/imagenet_family/imagenet-c'
        raise NotImplementedError


    elif args.dataset == 'TINY':
        root = os.path.join(data_path,'Tiny-ImageNet/tiny-imagenet-200/val')
        dataset_normal = datasets.ImageFolder(root, transform=test_transform)
        root = os.path.join(data_path,'Tiny-ImageNet-C-ImageFolder')
        print(root)

    dataset_brightness = datasets.ImageFolder(os.path.join(root, 'brightness'), transform=test_transform)
    dataset_contrast = datasets.ImageFolder(os.path.join(root, 'contrast'), transform=test_transform)
    dataset_defocus_blur = datasets.ImageFolder(os.path.join(root, 'defocus_blur'), transform=test_transform)
    dataset_elastic_transform = datasets.ImageFolder(os.path.join(root, 'elastic_transform'), transform=test_transform)
    dataset_fog = datasets.ImageFolder(os.path.join(root, 'fog'), transform=test_transform)
    dataset_frost = datasets.ImageFolder(os.path.join(root, 'frost'), transform=test_transform)
    dataset_gaussian_blur = datasets.ImageFolder(os.path.join(root, 'gaussian_blur'), transform=test_transform) if 'CIFAR' in args.dataset else None
    dataset_gaussian_noise = datasets.ImageFolder(os.path.join(root, 'gaussian_noise'), transform=test_transform)
    dataset_glass_blur = datasets.ImageFolder(os.path.join(root, 'glass_blur'), transform=test_transform)
    dataset_impulse_noise = datasets.ImageFolder(os.path.join(root, 'impulse_noise'), transform=test_transform)
    dataset_jpeg_compression = datasets.ImageFolder(os.path.join(root, 'jpeg_compression'), transform=test_transform)
    dataset_motion_blur = datasets.ImageFolder(os.path.join(root, 'motion_blur'), transform=test_transform)
    dataset_pixelate = datasets.ImageFolder(os.path.join(root, 'pixelate'), transform=test_transform)
    dataset_saturate = datasets.ImageFolder(os.path.join(root, 'saturate'), transform=test_transform) if 'CIFAR' in args.dataset else None
    dataset_shot_noise = datasets.ImageFolder(os.path.join(root, 'shot_noise'), transform=test_transform)
    dataset_snow = datasets.ImageFolder(os.path.join(root, 'snow'), transform=test_transform)
    dataset_spatter = datasets.ImageFolder(os.path.join(root, 'spatter'), transform=test_transform) if 'CIFAR' in args.dataset else None
    dataset_speckle_noise = datasets.ImageFolder(os.path.join(root, 'speckle_noise'), transform=test_transform) if 'CIFAR' in args.dataset else None
    dataset_zoom_blur = datasets.ImageFolder(os.path.join(root, 'zoom_blur'), transform=test_transform)

    return [dataset_normal, dataset_brightness, dataset_contrast, dataset_defocus_blur, dataset_elastic_transform, dataset_fog, dataset_frost, dataset_gaussian_blur, dataset_gaussian_noise, dataset_glass_blur, dataset_impulse_noise, dataset_jpeg_compression, dataset_motion_blur, dataset_pixelate, dataset_saturate, dataset_shot_noise, dataset_snow, dataset_spatter, dataset_speckle_noise, dataset_zoom_blur]

def build_transform(args):
    if args.dataset == 'CIFAR100':
        norm_train = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        norm_test = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    elif args.dataset == 'CIFAR10':
        norm_train = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        norm_test = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    elif args.dataset == 'ImageNet':
        norm_train = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        norm_test = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.dataset == 'TINY':
        norm_train = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        norm_test = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])

    train_transform= transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        # trainsforms.RandomHorizontalFlip(0.5),
        # transforms.RandomRotation(3),
        transforms.ToTensor(),
        norm_train])

    test_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm_test])

    return train_transform, test_transform