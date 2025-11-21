# Pneumonia Classification using ResNet-18

**Deep Learning-based Medical Image Classification: Pneumonia Detection from Chest X-rays**

## 📋 Assignment Overview

This assignment focuses on building and comparing two deep learning approaches for **pneumonia classification** using chest X-ray images. The project implements a ResNet-18 neural network architecture and compares training from scratch versus transfer learning, evaluating their effectiveness in automated pneumonia detection—a critical task in medical imaging.

### Why This Assignment?

Pneumonia classification is an important real-world application of deep learning in healthcare that demonstrates:

- **Medical AI Relevance**: Assists radiologists in pneumonia detection, potentially reducing diagnostic errors
- **Deep Learning Fundamentals**: Teaches CNN architecture, training strategies, and model optimization
- **Transfer Learning**: Compares the efficiency of pre-trained vs. from-scratch training approaches
- **Class Imbalance**: Addresses real-world challenges with imbalanced medical datasets (fewer Normal vs. more Pneumonia cases)
- **Model Evaluation**: Uses metrics like precision, recall, F1-score relevant to medical applications
- **Healthcare Impact**: Early pneumonia detection can significantly improve patient outcomes

---

## 🎯 Learning Objectives

By completing this assignment, you will:

1. **Understand ResNet-18 Architecture** - Learn residual networks and skip connections
2. **Train DNNs from Scratch** - Initialize and train models with random weights on medical images
3. **Apply Transfer Learning** - Fine-tune pre-trained ImageNet models for medical image classification
4. **Handle Class Imbalance** - Deal with unequal class distributions (234 Normal vs. 390 Pneumonia)
5. **Evaluate Medical AI Models** - Compute precision, recall, F1-score, and understand false positives/negatives
6. **Interpret Deep Learning Results** - Analyze misclassifications and model behavior
7. **Optimize Hyperparameters** - Tune learning rates, batch sizes, and epochs for better performance

---

## 📊 Dataset

### Chest X-Ray Images Dataset

| Metric | Value |
|--------|-------|
| **Total Images** | 5,857 |
| **Training Set** | 5,217 images |
| **Validation Set** | 16 images |
| **Test Set** | 624 images |
| **Image Size** | 224×224 pixels |
| **Normal Cases** | 234 test samples (40%) |
| **Pneumonia Cases** | 390 test samples (60%) |
| **Class Imbalance Ratio** | 1.66:1 |

### Data Augmentation

**For Training (from-scratch & transfer learning):**
- `RandomResizedCrop(224)` - Random crops and resizes
- `RandomHorizontalFlip` - Horizontal flips (realistic for X-rays)

**For Validation & Test:**
- `Resize(256)` + `CenterCrop(224)` - Center crop without augmentation

---

## 🔬 Two Approaches: Task Comparison

### **Task 1.1: Training ResNet-18 from Scratch**

Train a ResNet-18 model with **randomly initialized weights** on the pneumonia dataset.

#### Hyperparameters:

| Hyperparameter | Value |
|---|---|
| **Learning Rate** | 0.001 |
| **Batch Size** | 32 |
| **Training Epochs** | 20 |
| **Optimizer** | Adam |
| **Loss Function** | Cross Entropy Loss |
| **Hardware** | GPU (Colab) |

#### Results:

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | 75.00% (Epoch 19) |
| **Test Accuracy** | 89.74% |
| **Training Loss** | 0.2353 |
| **Validation Loss** | 0.5526 |

**Classification Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Normal** | 92% | 79% | 85% | 234 |
| **Pneumonia** | 89% | 96% | 92% | 390 |
| **Accuracy** | — | — | 90% | 624 |

#### Key Observations:
- ✅ **Strong Pneumonia Detection**: 96% recall (detects 96% of actual pneumonia cases)
- ⚠️ **Weak Normal Detection**: 79% recall (misses 21% of normal cases)
- 📊 **Model Bias**: Favors pneumonia detection, leading to over-sensitivity
- 🔴 **False Positives**: 8% of normal cases incorrectly flagged as pneumonia

---

### **Task 1.2: Training ResNet-18 with Transfer Learning (Pre-trained)**

Fine-tune a ResNet-18 model **pre-trained on ImageNet** weights for pneumonia classification.

#### Hyperparameters:

| Hyperparameter | Value |
|---|---|
| **Learning Rate** | 0.0001 (lower to prevent overfitting) |
| **Batch Size** | 32 |
| **Training Epochs** | 20 |
| **Optimizer** | Adam |
| **Loss Function** | Cross Entropy Loss |
| **Hardware** | GPU (Colab) |
| **Pre-training Dataset** | ImageNet |

#### Results:

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | 100% (some batches) |
| **Test Accuracy** | 95.19% ⭐ (Best) |
| **Training Loss** | Gradually decreasing |
| **Validation Loss** | Stable with minor fluctuations |

**Classification Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Normal** | 97% | 90% | 93% | 234 |
| **Pneumonia** | 94% | 98% | 96% | 390 |
| **Accuracy** | — | — | 95% | 624 |

#### Key Observations:
- ✅ **Excellent Overall Performance**: 95.19% accuracy vs. 89.74% from-scratch
- ✅ **Balanced Recall**: 90% normal detection, 98% pneumonia detection
- 📈 **Improvement**: +5.45% accuracy over from-scratch approach
- ✅ **Better generalization**: Transfer learning works significantly better

---

## 🎓 Running on Google Colab Pro

This assignment is designed to run on **Google Colab Pro** with GPU acceleration for faster training.

### ⚡ Colab Pro Setup Guide

#### Step 1: Open Google Colab

1. Go to https://colab.research.google.com/
2. Click **"File"** → **"New notebook"**
3. Click **Runtime** (top right) → **"Change runtime type"**
4. Select:
   - **Runtime type**: Python 3
   - **Hardware accelerator**: GPU (T4 or A100 if available with Pro)
5. Click **"Save"**

#### Step 2: Enable GPU and Check Resources

Copy and run this in a Colab cell:

```python
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("CUDA Version:", torch.version.cuda)
```

**Expected Output:**
```
GPU Available: True
GPU Name: Tesla T4 (or A100 with Colab Pro)
CUDA Version: 11.x
```

---

## 🚀 Installation & Setup in Colab

### Step 1: Install Required Packages

Run this in the first Colab cell:

```python
# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install scikit-learn matplotlib pandas numpy scipy Pillow tqdm tensorboard

# Verify installation
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
```

### Step 2: Mount Google Drive (Optional but Recommended)

To save your models and results:

```python
from google.colab import drive
drive.mount('/content/drive')

# Create working directory
import os
os.makedirs('/content/drive/My Drive/pneumonia_classification', exist_ok=True)
os.chdir('/content/drive/My Drive/pneumonia_classification')
```

### Step 3: Download Dataset from Kaggle

**Option A: Using Kaggle API (Recommended)**

```python
# Install Kaggle API
!pip install kaggle

# Upload kaggle.json (get from https://www.kaggle.com/settings/account)
from google.colab import files
files.upload()  # Upload kaggle.json

# Setup Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract dataset
!unzip -q chest-xray-pneumonia.zip
```

**Option B: Manual Download (If API fails)**

```python
# Download directly from web
!wget https://example.com/chest-xray-pneumonia.zip
!unzip -q chest-xray-pneumonia.zip
```

After download, verify structure:

```python
import os

# Check dataset structure
data_path = 'chest_xray'
for split in ['train', 'val', 'test']:
    for category in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(data_path, split, category)
        count = len(os.listdir(path))
        print(f"{split}/{category}: {count} images")
```

**Expected Output:**
```
train/NORMAL: 1349 images
train/PNEUMONIA: 3875 images
val/NORMAL: 8 images
val/PNEUMONIA: 8 images
test/NORMAL: 234 images
test/PNEUMONIA: 390 images
```

---

## 📝 Code Structure in Colab

### Complete Notebook Setup

Create a new Colab notebook with these sections:

```python
# ============================================
# SECTION 1: Setup & Imports
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # Use tqdm.notebook for Colab
import json

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# SECTION 2: Dataset Setup
# ============================================

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load Normal images (label=0)
        normal_dir = os.path.join(image_dir, 'NORMAL')
        for img in os.listdir(normal_dir):
            self.images.append(os.path.join(normal_dir, img))
            self.labels.append(0)
        
        # Load Pneumonia images (label=1)
        pneumonia_dir = os.path.join(image_dir, 'PNEUMONIA')
        for img in os.listdir(pneumonia_dir):
            self.images.append(os.path.join(pneumonia_dir, img))
            self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# No augmentation for validation/test
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ChestXrayDataset('chest_xray/train', transform=train_transform)
val_dataset = ChestXrayDataset('chest_xray/val', transform=val_transform)
test_dataset = ChestXrayDataset('chest_xray/test', transform=val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================
# SECTION 3: Task 1.1 - Train from Scratch
# ============================================

def train_from_scratch():
    # Create model with random initialization
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)  # 2 classes: Normal, Pneumonia
    model = model.to(device)
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    return model, train_losses, val_accuracies

# Run training from scratch
print("=" * 50)
print("TASK 1.1: Training ResNet-18 from Scratch")
print("=" * 50)
model_scratch, train_losses, val_accs = train_from_scratch()

# Save model
torch.save(model_scratch.state_dict(), 'resnet18_from_scratch.pth')
print("Model saved to: resnet18_from_scratch.pth")

# ============================================
# SECTION 4: Task 1.2 - Transfer Learning
# ============================================

def train_transfer_learning():
    # Load pre-trained ResNet-18
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)  # Replace final layer
    model = model.to(device)
    
    # Training parameters (lower learning rate for fine-tuning)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 20
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    return model, train_losses, val_accuracies

# Run transfer learning
print("\n" + "=" * 50)
print("TASK 1.2: Transfer Learning with Pre-trained ResNet-18")
print("=" * 50)
model_transfer, train_losses_tl, val_accs_tl = train_transfer_learning()

# Save model
torch.save(model_transfer.state_dict(), 'resnet18_transfer_learning.pth')
print("Model saved to: resnet18_transfer_learning.pth")

# ============================================
# SECTION 5: Evaluation on Test Set
# ============================================

def evaluate_model(model, test_loader, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n{model_name} - Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['Normal', 'Pneumonia']))
    
    return accuracy, all_preds, all_labels

print("\n" + "=" * 50)
print("EVALUATION ON TEST SET")
print("=" * 50)

acc_scratch, preds_scratch, labels_scratch = evaluate_model(model_scratch, test_loader, 
                                                              "From Scratch Model")
acc_transfer, preds_transfer, labels_transfer = evaluate_model(model_transfer, test_loader, 
                                                                "Transfer Learning Model")

# ============================================
# SECTION 6: Comparison & Visualization
# ============================================

print("\n" + "=" * 50)
print("RESULTS COMPARISON")
print("=" * 50)
print(f"From Scratch Accuracy: {acc_scratch*100:.2f}%")
print(f"Transfer Learning Accuracy: {acc_transfer*100:.2f}%")
print(f"Improvement: {(acc_transfer - acc_scratch)*100:.2f}%")

# Plot accuracies
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Validation Accuracy
axes[0].plot(range(1, epochs+1), val_accs, 'b-', label='From Scratch')
axes[0].plot(range(1, epochs+1), val_accs_tl, 'r-', label='Transfer Learning')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Accuracy (%)')
axes[0].set_title('Validation Accuracy Comparison')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_scratch = confusion_matrix(labels_scratch, preds_scratch)
cm_transfer = confusion_matrix(labels_transfer, preds_transfer)

axes[1].imshow(cm_transfer, cmap='Blues', interpolation='nearest')
axes[1].set_title('Transfer Learning - Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].colorbar(axes[1].images[0])

plt.tight_layout()
plt.savefig('comparison_results.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nPlot saved to: comparison_results.png")
```

---

## 🏃 Running the Code Step-by-Step in Colab

### Run Order in Colab Cells:

**Cell 1: Install packages**
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install scikit-learn matplotlib pandas numpy scipy Pillow tqdm tensorboard
```

**Cell 2: Mount Google Drive (Optional)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 3: Download dataset**
```python
!pip install kaggle
# Upload kaggle.json and download dataset
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q chest-xray-pneumonia.zip
```

**Cell 4-10: Run the complete code sections above**
- Section 1: Imports & GPU setup
- Section 2: Dataset creation
- Section 3: Task 1.1 training
- Section 4: Task 1.2 training
- Section 5: Evaluation
- Section 6: Visualization

---

## 📊 Expected Colab Runtime

| Task | Estimated Time on T4 GPU |
|---|---|
| Dataset download & setup | 5-10 min |
| Task 1.1 (From Scratch) | 20-25 min |
| Task 1.2 (Transfer Learning) | 18-22 min |
| Evaluation & Visualization | 2-3 min |
| **Total** | ~50-60 min |

---

## 💾 Saving Models & Results to Google Drive

After training:

```python
# Save models to Google Drive
import shutil

# Copy models
shutil.copy('resnet18_from_scratch.pth', 
            '/content/drive/My Drive/pneumonia_classification/resnet18_from_scratch.pth')
shutil.copy('resnet18_transfer_learning.pth', 
            '/content/drive/My Drive/pneumonia_classification/resnet18_transfer_learning.pth')

# Save results as JSON
results = {
    'from_scratch_accuracy': acc_scratch * 100,
    'transfer_learning_accuracy': acc_transfer * 100,
    'improvement': (acc_transfer - acc_scratch) * 100
}

with open('/content/drive/My Drive/pneumonia_classification/results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Models and results saved to Google Drive!")
```

---

## 📥 Downloading Results from Colab

After training completes:

1. **Download models**:
   ```python
   from google.colab import files
   files.download('resnet18_from_scratch.pth')
   files.download('resnet18_transfer_learning.pth')
   files.download('comparison_results.png')
   ```

2. **Download plots**:
   ```python
   files.download('comparison_results.png')
   ```

---

## ⚠️ Colab-Specific Tips

### Memory Management
If you run out of memory:

```python
# Clear cache between tasks
import gc
torch.cuda.empty_cache()
gc.collect()
```

### Timeout Prevention
Colab disconnects after inactivity. To keep it alive:

```python
# Run this in a cell to keep session active
from IPython.display import HTML
import time

def keep_alive():
    display(HTML('''
    <script>
    function ClickConnect(){
        console.log("Staying alive!");
        document.querySelector("colab-toolbar-button#connect").click()
    }
    setInterval(ClickConnect, 60000)
    </script>
    '''))

keep_alive()
```

### Hardware Selection
- **Colab Pro**: T4 GPU (medium speed, ~20 min per task)
- **Colab Pro+**: A100 GPU (fastest, ~10 min per task)

---
- **Assignment**: Assignment  - Pneumonia Classification
- **Institution**: University of Central Florida (UCF)
- **Platform**: Google Colab Pro
- **Semester**: Fall 2025

---

## 📚 References & Resources

- [Google Colab Official Guide](https://colab.research.google.com/)
- [PyTorch on Colab](https://pytorch.org/tutorials/)
- [ResNet Paper - Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [Chest X-Ray Dataset - Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Transfer Learning in PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

