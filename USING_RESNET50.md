# 🔍 Using ResNet50 for Car Damage Detection

## Understanding ResNet50

**ResNet50** (`resnet50-19c8e357.pth`) is a **backbone/feature extractor**, not a complete detection model. It extracts visual features from images but cannot detect damage by itself.

### What ResNet50 Does:
- ✅ Extracts visual features from images
- ✅ Serves as backbone for detection models
- ❌ Cannot detect objects/damage directly
- ❌ Needs detection head on top

---

## ✅ Your System Already Uses ResNet50!

### 1. **Mask R-CNN Model** (Primary ResNet50 Usage)

The Mask R-CNN model uses **ResNet50-FPN** as its backbone:

```python
# src/models/maskrcnn_model.py
model = maskrcnn_resnet50_fpn(
    pretrained=True,  # Uses ResNet50 pretrained on ImageNet
    pretrained_backbone=True
)
```

**Architecture:**
```
Input Image → ResNet50 Backbone → FPN → Detection Head → Damage Detections
                                      ↓
                                 Segmentation Head → Damage Masks
```

### 2. **Hybrid Model** (Optional ResNet50 Usage)

The hybrid model can use ResNet50 as encoder:

```python
# src/models/hybrid_model.py
segmentor = smp.UnetPlusPlus(
    encoder_name='resnet50',  # ResNet50 as encoder
    encoder_weights='imagenet',
    classes=num_classes
)
```

---

## 🎯 How to Use ResNet50 for Damage Detection

### Option 1: Train Mask R-CNN (Recommended - Uses ResNet50)

```powershell
# Train Mask R-CNN with ResNet50 backbone
python train.py --config config/config.yaml --model maskrcnn --epochs 100
```

**What happens:**
1. ✅ Loads ResNet50 pretrained weights (`resnet50-19c8e357.pth`)
2. ✅ Uses ResNet50 as feature extractor
3. ✅ Adds FPN (Feature Pyramid Network)
4. ✅ Adds detection and segmentation heads
5. ✅ Fine-tunes on car damage dataset

**Advantages:**
- 🎯 High precision detection
- 📐 Instance segmentation (exact damage shape)
- 🔬 Detailed feature extraction
- 📊 Better for complex damage patterns

**Disadvantages:**
- 🐌 Slower inference (~100-200ms per image)
- 💾 Larger model size (~180MB)
- 🔥 More GPU memory needed

### Option 2: Train Hybrid Model with ResNet50

```powershell
# Train hybrid model with ResNet50 encoder
python train.py --config config/config.yaml --model hybrid --epochs 100
```

Edit `config/config.yaml` to use ResNet50:

```yaml
model:
  architecture: 'hybrid'
  backbone: 'yolov8s'  # For detection part
  seg_encoder: 'resnet50'  # ResNet50 for segmentation
  seg_architecture: 'unetplusplus'
```

**Architecture:**
```
Input Image → YOLOv8 → Fast Detection Boxes
            ↓
            → ResNet50 U-Net → Precise Segmentation Masks
```

### Option 3: Use Pretrained ResNet50 Directly

If you have a custom `resnet50-19c8e357.pth` file:

```python
# Custom script to use ResNet50 as backbone
import torch
import torchvision.models as models

# Load pretrained ResNet50
resnet50 = models.resnet50(pretrained=False)
resnet50.load_state_dict(torch.load('resnet50-19c8e357.pth'))

# Add detection head on top
# ... (requires custom implementation)
```

---

## 📊 Model Comparison for Damage Detection

| Model | Backbone | Speed | Accuracy | Best For |
|-------|----------|-------|----------|----------|
| **YOLOv8** | CSPDarknet | ⚡⚡⚡ Fast (5-10ms) | Good | Real-time detection |
| **Mask R-CNN** | **ResNet50-FPN** | 🐌 Slow (100-200ms) | ⭐⭐⭐ Excellent | High precision |
| **Hybrid** | YOLO + ResNet50 | ⚡⚡ Medium (20-50ms) | ⭐⭐ Very Good | Balanced |

---

## 🚀 Quick Start with ResNet50

### Step 1: Train Mask R-CNN (Uses ResNet50)

```powershell
python train.py --config config/config.yaml --model maskrcnn --epochs 100
```

**Expected Training Time:**
- GPU: ~6-8 hours
- CPU: ~24-48 hours

### Step 2: Run Inference

```powershell
# Detect damage using ResNet50-based model
python inference.py --model checkpoints/best_model.pth --source test_image.jpg --output results/
```

### Step 3: Evaluate Performance

```powershell
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml
```

---

## 🔧 Configuration for ResNet50 Models

### Mask R-CNN Configuration

Edit `config/config.yaml`:

```yaml
model:
  architecture: 'maskrcnn'
  backbone: 'resnet50'
  pretrained: true
  pretrained_backbone: true
  trainable_backbone_layers: 3  # Fine-tune last 3 layers

training:
  epochs: 100
  batch_size: 4  # Reduce if out of memory
  learning_rate: 0.001
  optimizer: 'adamw'
  
dataset:
  num_classes: 6
  image_size: 800  # Mask R-CNN uses variable sizes
```

### Hybrid Model with ResNet50 Encoder

```yaml
model:
  architecture: 'hybrid'
  backbone: 'yolov8s'
  seg_encoder: 'resnet50'  # Use ResNet50 for segmentation
  seg_architecture: 'unetplusplus'
  pretrained: true

training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.001
```

---

## 💡 When to Use ResNet50-based Models

### Use Mask R-CNN (ResNet50) When:
✅ You need **highest accuracy**
✅ You need **instance segmentation** (exact damage shapes)
✅ You need **precise bounding boxes**
✅ Speed is not critical (<10 FPS acceptable)
✅ You have GPU with 8GB+ VRAM
✅ Use cases: Insurance claims, detailed assessments, quality control

### Use YOLOv8 When:
✅ You need **real-time detection** (>30 FPS)
✅ You need to run on **edge devices** (mobile, embedded)
✅ Speed is critical
✅ Approximate detection is sufficient
✅ Use cases: Mobile apps, real-time video, web apps

### Use Hybrid Model When:
✅ You need **balance** of speed and accuracy
✅ You want both detection and segmentation
✅ You have medium GPU (4-8GB VRAM)
✅ Use cases: Production systems, batch processing

---

## 🎯 Recommended Approach for Your Use Case

Based on your dataset (CarDD):

### For Maximum Accuracy (Recommended):
```powershell
# 1. Train Mask R-CNN with ResNet50 backbone
python train.py --config config/config.yaml --model maskrcnn --epochs 100

# 2. Evaluate on test set
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml

# 3. Use for inference
python inference.py --model checkpoints/best_model.pth --source test_image.jpg --output results/
```

### For Production Deployment:
```powershell
# 1. Train YOLOv8 for speed
python train.py --config config/config.yaml --model yolov8 --backbone yolov8s --epochs 100

# 2. Export to ONNX for deployment
python export.py --model checkpoints/best_model.pth --format onnx

# 3. Deploy web app
streamlit run app.py
```

---

## 📈 Expected Performance

### Mask R-CNN (ResNet50 backbone)

**Metrics (after 100 epochs):**
- mAP@0.5: ~0.70-0.80
- mAP@0.5:0.95: ~0.50-0.60
- Precision: ~0.75-0.85
- Recall: ~0.70-0.80

**Inference Speed:**
- GPU (RTX 3080): ~100ms per image
- CPU: ~2-3 seconds per image

### ResNet50 Feature Extraction Layers

```
Layer               Output Shape        Parameters
-------------------------------------------------------
conv1              (64, 112, 112)       9,408
layer1 (ResBlock1) (256, 56, 56)        215,808
layer2 (ResBlock2) (512, 28, 28)        1,219,584
layer3 (ResBlock3) (1024, 14, 14)       7,098,368
layer4 (ResBlock4) (2048, 7, 7)         14,964,736
-------------------------------------------------------
Total: 23,507,904 parameters
```

---

## 🔍 Verify ResNet50 Usage

Check if ResNet50 is being used:

```python
# In Python console or script
import torch
from src.models.maskrcnn_model import create_maskrcnn_model

# Create model
model = create_maskrcnn_model(num_classes=7, pretrained=True)

# Check backbone
print(model.model.backbone)
# Should show: ResNet with FPN

# Check if ResNet50 weights are loaded
for name, param in model.model.backbone.body.named_parameters():
    print(f"{name}: {param.shape}")
    break  # Just show first layer
```

---

## 📦 ResNet50 Weight File Location

PyTorch automatically downloads ResNet50 weights to:
```
Windows: C:\Users\<username>\.cache\torch\hub\checkpoints\resnet50-19c8e357.pth
Linux/Mac: ~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
```

If you have this file locally, you can:
1. ✅ Let PyTorch use it automatically (it will detect it)
2. ✅ Copy it to the cache location above
3. ✅ Load it manually in custom scripts

---

## 🎓 Summary

**ResNet50 (`resnet50-19c8e357.pth`) is:**
- ✅ A backbone feature extractor
- ✅ Already used in your Mask R-CNN model
- ✅ Can be used in hybrid model encoder
- ✅ Provides high-quality features for detection

**To use it for damage detection:**
```powershell
# Simply train Mask R-CNN (it uses ResNet50 automatically)
python train.py --config config/config.yaml --model maskrcnn --epochs 100
```

**Your system automatically:**
1. ✅ Downloads/loads ResNet50 weights
2. ✅ Uses it as backbone in Mask R-CNN
3. ✅ Fine-tunes it on car damage data
4. ✅ Saves the complete model to checkpoints

---

## 🚀 Next Steps

```powershell
# Start training with ResNet50-based Mask R-CNN
python train.py --config config/config.yaml --model maskrcnn --epochs 100

# Or use in web app (after training)
streamlit run app.py
```

The ResNet50 backbone will provide excellent feature extraction for accurate damage detection! 🎯
