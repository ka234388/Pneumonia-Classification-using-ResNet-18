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

<img width="646" height="370" alt="image" src="https://github.com/user-attachments/assets/3555e04d-821e-438d-ba3c-a22cca5f36fe" />

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

<img width="1000" height="700" alt="image" src="https://github.com/user-attachments/assets/ba8c327a-8adc-4c75-98c3-ac1f4f693215" />

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
<img width="646" height="370" alt="image" src="https://github.com/user-attachments/assets/383ce228-c260-4418-bb2d-b96fd2f220c6" />

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
<img width="1000" height="700" alt="image" src="https://github.com/user-attachments/assets/82935a7c-061f-400f-9e3d-000b90bf20b3" />

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

