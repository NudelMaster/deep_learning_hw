import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import itertools
import copy
from models import get_model
import utils

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy on the fly
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = running_loss / len(loader.dataset)
    acc = 100 * correct / total
    return avg_loss, acc

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader.dataset)
    acc = 100 * correct / total
    return avg_loss, acc

def train_model(model, train_set, val_set, config, device, num_workers=0):
    """
    Full training loop with validation and optional early stopping.
    
    Args:
        model: The PyTorch model
        train_set: DataLoader OR Dataset for training
        val_set: DataLoader OR Dataset for validation
        config: Dictionary containing 'batch_size', 'epochs', 'learning_rate', etc.
        device: Torch device
    """
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Auto-wrap Datasets in DataLoaders if needed
    batch_size = config.get('batch_size', 64)
    # Use config value if present, otherwise use the function argument
    num_workers = config.get('num_workers', num_workers)
    
    # Ensure reproducibility
    utils.set_seed(42)
    generator = torch.Generator().manual_seed(42)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              generator=generator, num_workers=num_workers,
                              worker_init_fn=utils.seed_worker)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                            generator=generator, num_workers=num_workers,
                            worker_init_fn=utils.seed_worker)

    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer based on config
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    opt_name = config.get('optimizer', 'adam').lower()
    
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Extract momentum, default to 0.9
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        
    # Optional scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=config.get('epochs', 30), 
            eta_min=1e-6
        )

    num_epochs = config.get('epochs', 30)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Early stopping variables
    patience = config.get('patience', 100) # Default to high number if not specified
    counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [], 
        'val_loss': [],
        'val_acc': []
    }
    
    # Check if we should compute training accuracy (only for final training, not grid search)
    compute_train_acc = config.get('compute_train_acc', True)

    print(f"Starting training for {num_epochs} epochs on {device}...")
    
    for epoch in range(1, num_epochs + 1):
        # Optimization: Calculate train_acc during training pass to avoid double iteration
        train_loss, train_acc_epoch = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        val_loss, val_acc = evaluate_accuracy(model, val_loader, device)
        
        if scheduler:
            scheduler.step()
            
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if compute_train_acc:
            history['train_acc'].append(train_acc_epoch)

        # Track best model based on accuracy (or loss if you prefer)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # counter = 0 # Reset early stopping if using it
        
        # Early stopping logic (based on loss usually)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print(f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    return history

def run_grid_search(model_name, train_dataset, val_dataset, param_grid, device, num_workers=0):
    """
    Runs a grid search over hyperparameters for a specific model architecture.
    
    Args:
        model_name: String name of the model (e.g., 'cnn', 'mobilenet') passed to get_model
        train_dataset: Dataset for training
        val_dataset: Dataset for validation
        param_grid: Dictionary of parameters to search (excluding model_name)
        device: Torch device
        num_workers: Number of workers for DataLoaders
    """
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Starting grid search for '{model_name}' with {len(combinations)} candidates...")
    
    results = []
    best_acc = 0.0
    best_config = None
    
    for params in tqdm(combinations):
        print(f"\nEvaluating Config: {params}")
        
        # Copy params to avoid mutation
        current_config = params.copy()
        
        # Disable compute_train_acc for grid search to save time
        current_config['compute_train_acc'] = False
        
        # Build model
        try:
            # Create fresh model instance
            # We explicitly pass num_classes=10 as it's standard for STL-10
            # Note: get_model is robust to extra kwargs
            model = get_model(model_name, num_classes=10, **current_config).to(device)
            print(f"Model {model_name} built successfully.")
        except Exception as e:
            print(f"Error building model {model_name} with config {current_config}: {e}")
            continue
            
        # Ensure training params are present
        if 'lr' in current_config and 'learning_rate' not in current_config:
            current_config['learning_rate'] = current_config['lr']
        
        # Train
        history = train_model(model, train_dataset, val_dataset, config=current_config, device=device, num_workers=num_workers)
        final_val_acc = max(history['val_acc'])
        
        results.append({
            'params': current_config,
            'best_val_acc': final_val_acc
        })
        
        if final_val_acc > best_acc:
            best_acc = final_val_acc
            best_config = current_config
            
    return results, best_config
