# 🚀 Complete Start-to-End Guide: Car Damage Detection System

This guide will walk you through the entire process from setup to deployment.

---

## 📋 Table of Contents

1. [Initial Setup](#1-initial-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Training Models](#3-training-models)
4. [Evaluating Models](#4-evaluating-models)
5. [Running Inference](#5-running-inference)
6. [Using the Streamlit Web App](#6-using-the-streamlit-web-app)
7. [Advanced Usage](#7-advanced-usage)

---

## 1. Initial Setup

### Step 1.1: Verify Python Environment

```powershell
# Check Python version (should be 3.8+)
python --version

# Activate virtual environment (if not already activated)
damage\Scripts\activate
```

### Step 1.2: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | findstr torch
pip list | findstr ultralytics
pip list | findstr streamlit
```

### Step 1.3: Verify Dataset Structure

```powershell
# Check dataset structure
tree /F CarDD_release
```

Expected structure:
```
CarDD_release/
├── CarDD_COCO/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   └── instances_test2017.json
│   ├── train2017/  (images)
│   ├── val2017/    (images)
│   └── test2017/   (images)
└── CarDD_SOD/
    ├── CarDD-TR/   (train)
    ├── CarDD-VAL/  (validation)
    └── CarDD-TE/   (test)
```

---

## 2. Dataset Preparation

### Step 2.1: Analyze Dataset

```powershell
# Run dataset analysis
python analyze_dataset.py
```

**Output**: Statistics about images, annotations, damage types

### Step 2.2: Verify Data Configuration

Check `config/config.yaml`:

```yaml
dataset:
  root_dir: "CarDD_release/CarDD_COCO"
  annotations:
    train: "CarDD_release/CarDD_COCO/annotations/instances_train2017.json"
    val: "CarDD_release/CarDD_COCO/annotations/instances_val2017.json"
    test: "CarDD_release/CarDD_COCO/annotations/instances_test2017.json"
```

### Step 2.3: Verify YOLO Configuration

Check `cardd.yaml`:

```yaml
path: CarDD_release/CarDD_COCO
train: train2017
val: val2017
test: test2017

nc: 6  # number of classes
names: ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']
```

---

## 3. Training Models

### Option A: Train YOLOv8 Model (Recommended - Fastest)

```powershell
# Train YOLOv8 small model (balanced speed/accuracy)
python train.py --config config/config.yaml --model yolov8 --epochs 100

# Or train different sizes:
# YOLOv8n (fastest): python train.py --config config/config.yaml --model yolov8 --backbone yolov8n
# YOLOv8s (balanced): python train.py --config config/config.yaml --model yolov8 --backbone yolov8s
# YOLOv8m (accurate): python train.py --config config/config.yaml --model yolov8 --backbone yolov8m
```

**Training Time**: 
- YOLOv8n: ~2-3 hours (GPU) / ~8-12 hours (CPU)
- YOLOv8s: ~3-4 hours (GPU) / ~12-16 hours (CPU)

**Output**: 
- Model saved to: `checkpoints/best_model.pth`
- Training logs: `outputs/train_YYYYMMDD_HHMMSS/`

### Option B: Train Mask R-CNN (High Precision)

```powershell
# Train Mask R-CNN
python train.py --config config/config.yaml --model maskrcnn --epochs 50
```

**Training Time**: ~6-8 hours (GPU) / ~24-48 hours (CPU)

### Option C: Train Hybrid Model (Best Quality)

```powershell
# Train hybrid YOLO + segmentation model
python train.py --config config/config.yaml --model hybrid --epochs 100
```

**Training Time**: ~8-12 hours (GPU) / ~48+ hours (CPU)

### Monitor Training

**Option 1: TensorBoard**
```powershell
# In a new terminal
tensorboard --logdir outputs/
# Open http://localhost:6006
```

**Option 2: Watch Logs**
```powershell
# View training progress
type outputs\train_YYYYMMDD_HHMMSS\training.log
```

---

## 4. Evaluating Models

### Step 4.1: Evaluate on Test Set

```powershell
# Evaluate trained model
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml

# Evaluate with custom confidence threshold
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml --conf-threshold 0.35
```

**Output**: COCO metrics (mAP, AP50, AP75, etc.)

### Step 4.2: Compare Multiple Models

```powershell
# Train multiple models first, then compare
python demo.py  # Run demo 5 (model comparison)
```

---

## 5. Running Inference

### Step 5.1: Single Image Prediction

```powershell
# Predict on a single image
python inference.py --model checkpoints/best_model.pth --source path/to/image.jpg --output outputs/predictions/

# View results
explorer outputs\predictions\
```

### Step 5.2: Batch Image Processing

```powershell
# Process entire folder
python inference.py --model checkpoints/best_model.pth --source CarDD_release/CarDD_COCO/test2017/ --output outputs/batch_results/

# Process with custom settings
python inference.py --model checkpoints/best_model.pth --source CarDD_release/CarDD_COCO/test2017/ --output outputs/batch_results/ --conf-threshold 0.3 --save-txt
```

### Step 5.3: Video Processing

```powershell
# Process video file
python inference.py --model checkpoints/best_model.pth --source path/to/video.mp4 --output outputs/video_results/
```

### Step 5.4: Webcam Real-time Detection

```powershell
# Use webcam (camera index 0)
python inference.py --model checkpoints/best_model.pth --source 0 --output outputs/webcam_results/
```

Press `q` to quit webcam mode.

---

## 6. Using the Streamlit Web App

### Step 6.1: Start the Web App

```powershell
# Launch Streamlit app
streamlit run app.py
```

**Browser opens automatically at**: `http://localhost:8501`

### Step 6.2: Load Model in Web UI

1. **Sidebar** → Model Path: `checkpoints/best_model.pth`
2. **Sidebar** → Config Path: `config/config.yaml`
3. Click **"🔄 Load Model"**
4. Wait for success message

### Step 6.3: Upload and Detect

1. Go to **"📸 Image Upload"** tab
2. Click **"Browse files"**
3. Select car image with damage
4. Click **"🔍 Detect Damage"**
5. View results on right side

### Step 6.4: Adjust Settings

- **Confidence Threshold**: Adjust slider (0.0-1.0)
- **Show Labels**: Toggle on/off
- **Show Confidence**: Toggle on/off

### Step 6.5: Download Results

- **📥 Download Image**: Get annotated PNG
- **📥 Download JSON**: Get detection report

### Step 6.6: View Analytics

1. Go to **"📊 Analytics"** tab
2. View damage distribution chart
3. View confidence scores chart
4. Check summary statistics

---

## 7. Advanced Usage

### Export Model to ONNX

```powershell
# Export for deployment
python export.py --model checkpoints/best_model.pth --format onnx

# Export to multiple formats
python export.py --model checkpoints/best_model.pth --format onnx,torchscript,tflite
```

**Output**: Exported models in `exports/` folder

### Run Demos

```powershell
# Run all demo examples
python demo.py

# This will demonstrate:
# - Single image detection
# - Batch processing
# - Video processing
# - Custom visualization
# - Model comparison
# - Ensemble prediction
```

### Custom Training Configuration

Edit `config/config.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
```

Then train:
```powershell
python train.py --config config/config.yaml
```

### Resume Training

```powershell
# Resume from checkpoint
python train.py --config config/config.yaml --resume checkpoints/last_model.pth
```

---

## 🎯 Quick Start Workflow (Minimal Steps)

If you want the **fastest path** to get results:

```powershell
# 1. Verify setup
pip install -r requirements.txt

# 2. Analyze dataset
python analyze_dataset.py

# 3. Train quick model (YOLOv8n - smallest/fastest)
python train.py --config config/config.yaml --model yolov8 --backbone yolov8n --epochs 50

# 4. Run demo
python demo.py

# 5. Start web app
streamlit run app.py
```

---

## 📊 Expected Results

### Training Metrics (YOLOv8s, 100 epochs)
- **mAP@0.5**: ~0.65-0.75
- **mAP@0.5:0.95**: ~0.45-0.55
- **Precision**: ~0.70-0.80
- **Recall**: ~0.65-0.75

### Inference Speed
- **YOLOv8n**: ~5-10ms per image (GPU)
- **YOLOv8s**: ~10-15ms per image (GPU)
- **Mask R-CNN**: ~100-200ms per image (GPU)

---

## 🔧 Troubleshooting

### Issue: Training is slow

**Solution**:
- Use smaller model: YOLOv8n instead of YOLOv8l
- Reduce batch size in config
- Enable GPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Issue: Out of memory error

**Solution**:
```yaml
# Edit config/config.yaml
training:
  batch_size: 4  # Reduce from 16 to 4 or 8
```

### Issue: No detections in results

**Solution**:
- Lower confidence threshold: `--conf-threshold 0.15`
- Check if model trained properly
- Verify image quality

### Issue: Model file not found

**Solution**:
```powershell
# Check if checkpoint exists
dir checkpoints\

# Use correct path
python inference.py --model checkpoints/best_model.pth
```

### Issue: Streamlit won't start

**Solution**:
```powershell
# Reinstall streamlit
pip install --upgrade streamlit

# Clear cache
streamlit cache clear
```

---

## 📁 Key Files & Directories

| Path | Description |
|------|-------------|
| `config/config.yaml` | Main configuration file |
| `cardd.yaml` | YOLO dataset configuration |
| `train.py` | Training script |
| `inference.py` | Inference/prediction script |
| `evaluate.py` | Model evaluation script |
| `app.py` | Streamlit web application |
| `demo.py` | Demo examples |
| `checkpoints/` | Saved model weights |
| `outputs/` | Training logs and results |
| `CarDD_release/` | Dataset directory |

---

## 🎓 Learning Resources

1. **Dataset Analysis**: Run `python analyze_dataset.py` first
2. **Quick Demo**: Run `python demo.py` for examples
3. **Documentation**: Read `README.md` for details
4. **Web Interface**: Use `streamlit run app.py` for GUI

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and extracted to `CarDD_release/`
- [ ] Configuration files verified (`config/config.yaml`, `cardd.yaml`)
- [ ] Model trained (or pretrained weights downloaded)
- [ ] Inference tested on sample images
- [ ] Streamlit app running

---

## 🚀 Production Deployment

### Deploy Web App to Cloud

```powershell
# Push to GitHub
git add .
git commit -m "Add car damage detection system"
git push origin main

# Deploy to Streamlit Cloud
# 1. Go to share.streamlit.io
# 2. Connect your GitHub repo
# 3. Select app.py
# 4. Deploy!
```

### API Server (Optional)

Create FastAPI server:
```python
# api.py
from fastapi import FastAPI, UploadFile
from inference import CarDDInference

app = FastAPI()
inferencer = CarDDInference("checkpoints/best_model.pth", "config/config.yaml")

@app.post("/predict")
async def predict(file: UploadFile):
    image = await file.read()
    results = inferencer.predict_image(image)
    return results
```

Run:
```powershell
pip install fastapi uvicorn
uvicorn api:app --reload
```

---

## 💡 Tips for Best Results

1. **Train longer**: 100+ epochs for best accuracy
2. **Use augmentation**: Already enabled in config
3. **Ensemble models**: Combine YOLOv8 + Mask R-CNN
4. **Fine-tune threshold**: Test different confidence thresholds
5. **Use GPU**: ~10x faster training and inference

---

**Need Help?** 
- 📖 Check `README.md` for detailed documentation
- 📚 See `QUICKSTART.md` for 5-minute guide
- 🔧 Review `PROJECT_SUMMARY.md` for architecture overview

---

**Happy Detecting! 🚗💥🔍**
