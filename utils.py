import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    """
    Worker init function for DataLoader to ensure each worker is seeded differently but deterministically.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def plot_training_history(history):
    """
    Plots training and validation loss and accuracy curves.
    """
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    val_acc = history.get('val_acc', [])
    # train_acc might be missing
    train_acc = history.get('train_acc', [])
    
    epochs_range = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(epochs_range, train_loss, label='Train Loss')
    if val_loss:
        ax1.plot(epochs_range, val_loss, label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.grid(True)
    ax1.legend()
    
    # Accuracy plot
    if val_acc or train_acc:
        if train_acc:
            ax2.plot(epochs_range, train_acc, label='Train Acc')
        if val_acc:
            ax2.plot(epochs_range, val_acc, label='Val Acc')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.grid(True)
        ax2.legend()
        
    plt.tight_layout()
    plt.show()
