"""
Author: Tomer Porat

Utility functions for the pipeline
Dataset-agnostic: works with CIFAR10, MNIST, Fashion-MNIST, RIAWELC
"""

import torch
import os
from torchvision import datasets, transforms
from datasets_custom import RIAWELCDataset


def get_device():
    """Get the appropriate device (GPU/MPS/CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model, path, M, phase, epoch):
    """Save model checkpoint"""
    create_directory(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"✓ Saved checkpoint: {path}")


def load_checkpoint(model, path, device):
    """Load model checkpoint"""
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def get_compression_ratio(M, D):
    """Calculate compression ratio"""
    return D / M


def get_dataset_and_transforms(dataset_name, data_dir="./data"):
    """
    Get dataset class and preprocessing transforms based on dataset name

    All datasets resized to 32×32 for unified pipeline across:
    - CIFAR10 (native 32×32)
    - MNIST/FashionMNIST (upsampled from 28×28)
    - RIAWELC (downsampled from 224×224)

    Args:
        dataset_name: "cifar10", "mnist", "fmnist", or "riawelc"
        data_dir: data directory

    Returns:
        (dataset_class, train_transform, test_transform)
    """
    from config import IMAGE_SIZE, CIFAR10_AUGMENTATION, CIFAR10_GRAYSCALE, RIAWELC_FLIP_AUGMENTATION
    img_size = (IMAGE_SIZE, IMAGE_SIZE)  # 32×32 for all datasets

    if dataset_name == "riawelc":
        riawelc_train_ops = [transforms.Resize(img_size)]  # 224×224 → 32×32
        if RIAWELC_FLIP_AUGMENTATION:
            riawelc_train_ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        riawelc_train_ops += [
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
        return (
            RIAWELCDataset,
            transforms.Compose(riawelc_train_ops),
            transforms.Compose([
                transforms.Resize(img_size),  # 224×224 → 32×32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    elif dataset_name == "cifar10":
        cifar_norm_mean = (0.5,) if CIFAR10_GRAYSCALE else (0.5, 0.5, 0.5)
        cifar_norm_std = (0.5,) if CIFAR10_GRAYSCALE else (0.5, 0.5, 0.5)
        cifar_train_ops = [transforms.Resize(img_size)]  # Already 32×32, no-op
        if CIFAR10_GRAYSCALE:
            cifar_train_ops.append(transforms.Grayscale(num_output_channels=1))
        if CIFAR10_AUGMENTATION:
            # Standard CIFAR10 recipe: crop-with-padding + flip gives much
            # stronger regularization than the previous 5° affine (~+2-3%)
            cifar_train_ops += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        cifar_train_ops += [
            transforms.ToTensor(),
            transforms.Normalize(cifar_norm_mean, cifar_norm_std)
        ]
        cifar_test_ops = [transforms.Resize(img_size)]  # Already 32×32, no-op
        if CIFAR10_GRAYSCALE:
            cifar_test_ops.append(transforms.Grayscale(num_output_channels=1))
        cifar_test_ops += [
            transforms.ToTensor(),
            transforms.Normalize(cifar_norm_mean, cifar_norm_std)
        ]
        return (
            datasets.CIFAR10,
            transforms.Compose(cifar_train_ops),
            transforms.Compose(cifar_test_ops)
        )
    elif dataset_name == "mnist":
        return (
            datasets.MNIST,
            transforms.Compose([
                transforms.Resize(img_size),  # 28×28 → 32×32
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            transforms.Compose([
                transforms.Resize(img_size),  # 28×28 → 32×32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    elif dataset_name == "fmnist":
        return (
            datasets.FashionMNIST,
            transforms.Compose([
                transforms.Resize(img_size),  # 28×28 → 32×32
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            transforms.Compose([
                transforms.Resize(img_size),  # 28×28 → 32×32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
