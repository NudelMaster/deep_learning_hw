# Deep Learning Homework: STL-10 Experiments

This repository contains implementations of various neural network architectures for the STL-10 dataset, ranging from simple logistic regression to custom CNNs and transfer learning with MobileNetV2.

## Project Structure

The project is refactored into modular components for clarity and reusability:

*   **`deep_learning_hw3.ipynb`**: The main driver notebook. Use this to configure and run all experiments (Grid Search & Final Training).
*   **`models.py`**: Model definitions (`CNN`, `MobileNetFeatureExtractor`, etc.) and a factory function `get_model` to instantiate them.
*   **`trainer.py`**: Contains the core training loop (`train_model`), evaluation logic, and the automated `run_grid_search` engine.
*   **`data_setup.py`**: Handles STL-10 data downloading, splitting (Train/Val/Test), and applying appropriate transformations (Augmentation vs Clean).
*   **`utils.py`**: Visualization utilities for plotting loss/accuracy curves (`plot_training_history`) and displaying image batches.

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
# OR manually:
pip install torch torchvision numpy matplotlib scikit-learn tqdm jupyter
```

### 3. Data
The STL-10 dataset will be automatically downloaded to the `./data` directory when you first run the notebook or `data_setup.py`.

## Running the Experiments

1.  **Start Jupyter Notebook**:
    ```bash
    jupyter notebook deep_learning_hw3.ipynb
    ```
2.  **Follow the Notebook Flow**:
    The notebook is structured to run experiments sequentially for each model type.
    *   **Grid Search**: Automatically tests combinations of hyperparameters (LR, Batch Size, Optimizer, etc.) on a validation split.
    *   **Final Training**: Uses the best configuration found to train the model fully (more epochs, early stopping).
    *   **Visualization**: Plots training/validation loss and accuracy curves.
    *   **Testing** : Perform inference on test data and test accuracy

## Experiment Workflow

This repository is designed to run experiments in a two-stage process: **Grid Search** followed by **Final Training**.

### Step 1: Grid Search
Run `trainer.run_grid_search` to explore hyperparameter combinations.
- **Goal:** Find the best configuration (learning rate, batch size, etc.) based on validation accuracy.
- **Output:** A list of results and the `best_config` dictionary.
- **Note:** Grid search runs for fewer epochs (e.g., 5) to save time and disables training accuracy computation for speed.
- **Important:** `train_subset` is instantiated once in the "Global Setup" phase using `data_setup.get_stl10_splits()` and passed to the grid search function.

### Step 2: Final Training
Take the `best_config` from Step 1 and run `trainer.train_model`.
- **Goal:** Train the optimal model fully to convergence.
- **Modification:** Increase `epochs` (e.g., to 30) and set `patience` for early stopping.
- **Output:** Final training history (Loss/Accuracy plots) and a saved model.

### Step 3: Testing
Evaluate the trained model on the held-out test set (`test_ds`) to get the final performance metric.
- **Goal:** Verify generalization on unseen data.
- **Method:** Run inference in `eval` mode with `torch.no_grad()`.

### Supported Models
1.  **Logistic Regression** (`logreg`): Simple baseline (Input flattened).
2.  **Neural Network** (`nn`): Fully connected network (Input flattened).
3.  **CNN** (`cnn`): Custom convolutional network with scalable width.
4.  **MobileNet Frozen** (`mobilenet` with `freeze_backbone=True`): Feature extraction using pre-trained weights.
5.  **MobileNet Fine-Tune** (`mobilenet` with `freeze_backbone=False`): Full training of the backbone.

### Example Usage (in Notebook)

```python
import trainer
import models
import data_setup
import torch

# 1. Setup Data
train_subset, val_subset, test_ds = data_setup.get_stl10_splits()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define Grid
grid = {
    # Note: 'model_name' is passed separately, not in the grid
    'base_width': [16, 32],
    'learning_rate': [1e-3, 1e-4],
    'optimizer': ['adam'],
    'epochs': [5]
}

# 3. Run Search
results, best_config = trainer.run_grid_search(
    model_name='cnn',
    train_dataset=train_subset,
    val_dataset=val_subset,
    param_grid=grid,
    device=device
)

# 4. Train Best Model
best_config['epochs'] = 30
best_config['patience'] = 5
history = trainer.train_model(
    model=models.get_model('cnn', num_classes=10, **best_config).to(device),
    train_loader=train_subset, # Trainer auto-wraps in DataLoader
    val_loader=val_subset,     # Trainer auto-wraps in DataLoader
    config=best_config,
    device=device
)

# 5. Test Model
model.eval()
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
correct = 0
total = 0

print("Evaluating on Test Set...")
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Accuracy: {100 * correct / total:.2f} %')
```
