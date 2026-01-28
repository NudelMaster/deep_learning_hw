import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import copy

def get_stl10_splits(data_root='./data', train_val_split=0.8, download=True, train_dataset=None, test_dataset=None):
    """
    Returns the train, validation, and test splits for STL-10.
    
    The Training split (80% of original train) uses data augmentation (RandomCrop, Flip, etc.).
    The Validation split (20% of original train) uses deterministic transforms (CenterCrop).
    The Test split (original test set) uses deterministic transforms (CenterCrop).
    
    Returns:
        train_subset (Subset): Augmented training data
        val_subset (Subset): Clean validation data
        test_dataset (Dataset): Clean test data
    """
    
    # 1. Get base training dataset
    if train_dataset is not None:
        full_train_ds_aug = train_dataset
    else:
        full_train_ds_aug = datasets.STL10(root=data_root, split='train', download=download)

    # 2. Calculate mean/std based on training data
    data = full_train_ds_aug.data / 255.0
    train_mean = data.reshape(3, -1).mean(axis=1)
    train_std = data.reshape(3, -1).std(axis=1)
    
    # 3. Define Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])

    eval_transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])
        
    # 4. Setup Datasets
    # Apply train transform to the base dataset
    full_train_ds_aug.transform = train_transform
    
    # Create a shallow copy for validation
    full_train_ds_clean = copy.copy(full_train_ds_aug)
    full_train_ds_clean.transform = eval_transform
    
    # 5. Create indices for splitting
    num_train = len(full_train_ds_aug)
    train_size = int(num_train * train_val_split)
    
    # Use a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_train, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 6. Create Subsets
    train_subset = Subset(full_train_ds_aug, train_indices)
    val_subset = Subset(full_train_ds_clean, val_indices)
    
    # 7. Setup Test Dataset
    if test_dataset is not None:
        test_dataset_clean = test_dataset
    else:
        test_dataset_clean = datasets.STL10(root=data_root, split='test', download=download)
    
    test_dataset_clean.transform = eval_transform
    
    return train_subset, val_subset, test_dataset_clean
