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
| **Hardware** | GPU |

#### Architecture:
- ResNet-18 with 2 fully connected layers for binary classification
- Input: 224×224 chest X-ray images
- Output: 2 classes (Normal, Pneumonia)

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
- ⚠️ **Weak Normal Detection**: 79% recall (misses 21% of normal cases = false positives)
- 📊 **Model Bias**: Favors pneumonia detection, leading to over-sensitivity
- 🔴 **False Positives**: 8% of normal cases incorrectly flagged as pneumonia

#### Issue: Class Imbalance
- Pneumonia cases (390) outnumber Normal (234) by 1.66×
- Model learns to predict Pneumonia more frequently
- Solution: Use upsampling, class weights, or balanced sampling

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
| **Hardware** | GPU |
| **Pre-training Dataset** | ImageNet |

#### Architecture:
- ResNet-18 pre-trained on ImageNet
- Freeze early layers (already learned general features)
- Replace final layer with 2-class classifier
- Fine-tune with conservative augmentation

#### Results:

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | 100% (some batches reached perfection) |
| **Test Accuracy** | 95.19% ⭐ (Best) |
| **Training Loss** | Gradually decreasing |
| **Validation Loss** | Fluctuating (unstable but still improving) |

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
- ⚠️ **Validation Loss Fluctuations**: Spikes at epochs 2, 10, 11, 16 indicate some training instability
- 🔴 **Remaining Issues**: 10 false negatives for normal cases (over-sensitivity to pneumonia)

#### Why Transfer Learning Works Better:
1. **Pre-learned Features**: ImageNet weights already capture general image features (edges, textures, shapes)
2. **Faster Convergence**: Requires fewer epochs to achieve high accuracy
3. **Lower Learning Rate**: 0.0001 prevents catastrophic forgetting of learned features
4. **Better Generalization**: Reduces overfitting on limited medical dataset

---

## 🛠️ Requirements

### System Requirements
- Python 3.7+
- PyTorch/TorchVision with CUDA support (for GPU acceleration)
- 8GB+ RAM
- NVIDIA GPU (strongly recommended for faster training)
- 5GB+ disk space for dataset and model checkpoints

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies:
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
pandas>=1.2.0
Pillow>=8.0.0
tqdm
tensorboard
```

---

## 🚀 Installation & Setup Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/PneumoniaClassification.git
cd PneumoniaClassification
```

### Step 2: Create Virtual Environment

```bash
# Using Python venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Download the Chest X-Ray Images dataset:

```bash
# Option 1: Manual download from Kaggle
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Option 2: Programmatic download
python scripts/download_dataset.py
```

**Expected directory structure:**
```
data/
├── train/
│   ├── NORMAL/          # 1349 normal X-rays
│   └── PNEUMONIA/       # 3875 pneumonia X-rays
├── val/
│   ├── NORMAL/          # 8 normal X-rays
│   └── PNEUMONIA/       # 8 pneumonia X-rays
└── test/
    ├── NORMAL/          # 234 normal X-rays
    └── PNEUMONIA/       # 390 pneumonia X-rays
```

### Step 5: Organize Project Structure

```bash
mkdir -p outputs logs checkpoints results
```

---

## 🏃 How to Run the Code

### Task 1.1: Train ResNet-18 from Scratch

```bash
python train_from_scratch.py \
    --data_dir data/ \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --device cuda \
    --output_dir outputs/task1_1
```

**Arguments:**
- `--data_dir`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--device`: 'cuda' for GPU or 'cpu' for CPU
- `--output_dir`: Directory to save results

**Output:**
- Trained model saved to `checkpoints/resnet18_from_scratch.pth`
- Training logs and plots in `outputs/task1_1/`
- Test accuracy and metrics in `results/task1_1_results.json`

### Task 1.2: Train ResNet-18 with Transfer Learning

```bash
python train_transfer_learning.py \
    --data_dir data/ \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.0001 \
    --pretrained \
    --device cuda \
    --output_dir outputs/task1_2
```

**Arguments:**
- `--pretrained`: Use ImageNet pre-trained weights (default: True)
- `--lr`: Learning rate (default: 0.0001, lower for fine-tuning)
- Other arguments same as Task 1.1

**Output:**
- Trained model saved to `checkpoints/resnet18_pretrained.pth`
- Training logs and plots in `outputs/task1_2/`
- Test accuracy and metrics in `results/task1_2_results.json`

### Evaluate Both Models

```bash
python evaluate.py \
    --model1 checkpoints/resnet18_from_scratch.pth \
    --model2 checkpoints/resnet18_pretrained.pth \
    --data_dir data/test/ \
    --output_dir results/
```

### Generate Comparison Report

```bash
python compare_models.py \
    --results_dir results/ \
    --output_file comparison_report.txt
```

### Visualize Results

```bash
python visualize_results.py \
    --results_dir results/ \
    --output_dir plots/
```

---

## 📂 Project Structure

```
PneumoniaClassification/
├── data/                          # Dataset directory
│   ├── train/                     # Training images (5217 total)
│   ├── val/                       # Validation images (16 total)
│   └── test/                      # Test images (624 total)
├── src/                           # Source code
│   ├── train_from_scratch.py     # Task 1.1 training script
│   ├── train_transfer_learning.py # Task 1.2 training script
│   ├── evaluate.py               # Model evaluation
│   ├── compare_models.py         # Compare two approaches
│   ├── models.py                 # ResNet-18 model definition
│   └── utils.py                  # Utility functions
├── scripts/                       # Helper scripts
│   ├── download_dataset.py       # Download Kaggle dataset
│   └── visualize_results.py      # Generate plots
├── checkpoints/                   # Saved model weights
│   ├── resnet18_from_scratch.pth
│   └── resnet18_pretrained.pth
├── outputs/                       # Training logs and plots
│   ├── task1_1/                  # From-scratch results
│   └── task1_2/                  # Transfer learning results
├── results/                       # Final results and metrics
│   ├── task1_1_results.json
│   ├── task1_2_results.json
│   └── comparison_report.txt
├── requirements.txt              # Python dependencies
├── annotated-Report_Assignment1.pdf  # Detailed report
└── README.md                     # This file
```

---

## 📊 Results Summary & Comparison

### Performance Metrics Comparison

| Metric | From Scratch (Task 1.1) | Transfer Learning (Task 1.2) | Improvement |
|---|---|---|---|
| **Test Accuracy** | 89.74% | 95.19% | +5.45% ⭐ |
| **Normal Precision** | 92% | 97% | +5% |
| **Normal Recall** | 79% | 90% | +11% |
| **Pneumonia Precision** | 89% | 94% | +5% |
| **Pneumonia Recall** | 96% | 98% | +2% |
| **Training Time** | ~30 min | ~25 min | Faster |
| **Validation Loss** | Stable | Fluctuating | More stable (scratch) |

### Key Findings

**Transfer Learning Advantages:**
- ✅ **5.45% higher accuracy** (95.19% vs. 89.74%)
- ✅ **Better normal case detection** (+11% recall improvement)
- ✅ **Fewer false positives** - More balanced precision-recall
- ✅ **Better for medical AI** - Critical to reduce false positives in healthcare

**From Scratch Advantages:**
- ✅ More stable validation loss (no fluctuations)
- ✅ Lower overfitting indicators
- ✅ Demonstrates full learning process

**Conclusion:** **Transfer Learning is Recommended** for pneumonia classification due to:
1. Significantly higher accuracy (95% vs. 90%)
2. Better balanced detection (90% normal, 98% pneumonia)
3. More suitable for clinical deployment
4. Better handles limited medical image dataset

---

## 🏥 Medical Implications

### Normal Case Analysis (Task 1.1)
- **Recall: 79%** → 21 out of 100 normal cases misclassified as pneumonia
- **Risk**: False positives lead to unnecessary treatment
- **Impact**: Patient anxiety, additional procedures, healthcare costs

### Normal Case Analysis (Task 1.2)
- **Recall: 90%** → Only 10 out of 100 normal cases misclassified
- **Improvement**: 11% fewer false positives
- **Impact**: Better patient outcomes, reduced unnecessary intervention

### Pneumonia Case Analysis (Task 1.2)
- **Recall: 98%** → Only 2 misclassified pneumonia cases per 100
- **Critical**: Very few missed diagnoses (high sensitivity)
- **Impact**: Early treatment, better prognosis

---

## ⚠️ Common Issues & Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python train_transfer_learning.py --batch_size 16

# Or use CPU
python train_transfer_learning.py --device cpu
```

### Issue 2: Low Test Accuracy

**Causes**: Overfitting, wrong learning rate, class imbalance

**Solutions**:
```bash
# Try lower learning rate
python train_transfer_learning.py --lr 0.00005

# Use class weights for imbalance
python train_transfer_learning.py --use_class_weights
```

### Issue 3: Validation Loss Spikes

**Cause**: Learning rate too high, unstable training

**Solution**:
```bash
# Reduce learning rate
python train_transfer_learning.py --lr 0.00005

# Use learning rate scheduler
python train_transfer_learning.py --use_scheduler
```

### Issue 4: Dataset Not Found

**Error**: `FileNotFoundError: data directory not found`

**Solution**:
```bash
# Download dataset
python scripts/download_dataset.py

# Or manually place data in correct structure
mkdir -p data/train/NORMAL data/train/PNEUMONIA
mkdir -p data/val/NORMAL data/val/PNEUMONIA
mkdir -p data/test/NORMAL data/test/PNEUMONIA
```

---

## 🔬 Technical Concepts

### ResNet-18 Architecture

ResNet-18 introduces **residual connections** (skip connections):

```
Input → Conv → [Residual Block] → [Residual Block] → ... → FC (2 classes)
         ↓                            ↓
         └──────────────────────────┘ (skip connection)
```

**Benefits:**
- Allows training of very deep networks
- Prevents vanishing gradient problem
- Learns residual (difference) rather than full transformation

### Transfer Learning

Pre-trained ImageNet weights provide:
- Edge detection filters (layer 1-2)
- Texture and pattern recognition (layer 3-4)
- High-level object features (layer 5-6)

**Fine-tuning strategy:**
- Freeze early layers (keep general features)
- Train final layers on medical images
- Use low learning rate to preserve pre-trained knowledge

### Class Imbalance Handling

**Problem**: 390 pneumonia vs. 234 normal cases

**Solutions:**
1. **Class Weights**: Weight loss by inverse class frequency
2. **Upsampling**: Duplicate minority (normal) samples
3. **Downsampling**: Remove majority (pneumonia) samples
4. **Balanced Sampling**: Custom data loader with balanced batches

---

## 🎓 Course Information

- **Course**: CAP 6411 - Computer Vision Systems
- **Assignment**: Assignment 1 - Pneumonia Classification
- **Institution**: University of Central Florida (UCF)
- **Semester**: Fall 2025

---

## 📚 References & Resources

- [ResNet Paper - Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transfer Learning in PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ImageNet Pre-trained Models](https://pytorch.org/vision/stable/models.html)
- [Chest X-Ray Dataset - Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Class Imbalance in Medical Imaging](https://arxiv.org/abs/1811.02521)
- [Scikit-learn Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 💡 Recommendations & Best Practices

1. **For Clinical Deployment**:
   - Use **Task 1.2 (Transfer Learning)** model (95.19% accuracy)
   - Implement confidence thresholding to reduce false positives
   - Use ensemble methods combining multiple models
   - Always require radiologist confirmation (AI as assistance, not replacement)

2. **To Improve Model Further**:
   - Apply class balancing (upsampling normal cases)
   - Use ensemble of multiple architectures
   - Implement data augmentation (rotation, zoom, elastic deformation)
   - Try deeper architectures (ResNet-50, ResNet-101)
   - Apply Grad-CAM for interpretability

3. **For Medical AI Safety**:
   - Always validate on independent test set
   - Monitor false positive rate (critical for normal cases)
   - Consider domain shift (different hospitals, equipment)
   - Implement continuous performance monitoring

---

**Last Updated**: November 21, 2025  
**Repository**: https://github.com/yourusername/PneumoniaClassification

## License
This project is for academic purposes as part of CAP 6411 course at UCF.
