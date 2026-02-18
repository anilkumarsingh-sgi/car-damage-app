# ⚠️ Common Errors and Solutions

## Error 1: ImportError - Relative Import with No Parent Package

### Error Message:
```
ImportError: attempted relative import with no known parent package
File: src/models/hybrid_model.py
Line: from .yolo_model import YOLOv8CarDD
```

### Cause:
You tried to run `hybrid_model.py` directly as a script:
```powershell
python src/models/hybrid_model.py  # ❌ WRONG
```

### Explanation:
`hybrid_model.py` is a **module**, not a **script**. It uses relative imports (`.yolo_model`) which only work when imported as part of a package, not when run directly.

### Solution:
**Don't run model files directly!** Instead, use the provided scripts:

```powershell
# ✅ Correct ways to use the models:

# 1. Train a model
python train.py --config config/config.yaml --model hybrid

# 2. Run inference
python inference.py --model checkpoints/best_model.pth --source image.jpg

# 3. Evaluate model
python evaluate.py --model checkpoints/best_model.pth

# 4. Use web app
streamlit run app.py
```

---

## Error 2: ModuleNotFoundError - prettytable

### Error Message:
```
ModuleNotFoundError: No module named 'prettytable'
```

### Cause:
Missing dependency that wasn't in the original `requirements.txt`

### Solution:
```powershell
# ✅ Install prettytable
pip install prettytable

# Or reinstall all dependencies
pip install -r requirements.txt
```

### Status:
✅ **FIXED** - prettytable is now installed and `analyze_dataset.py` works!

---

## How to Use Each Component

### 📊 Dataset Analysis
```powershell
python analyze_dataset.py
```
**Purpose**: Analyze dataset statistics and visualizations
**Output**: 
- Console statistics
- `dataset_analysis_train.png`
- `dataset_analysis_val.png`
- `dataset_analysis_test.png`

### 🏋️ Model Training
```powershell
# YOLOv8 (recommended)
python train.py --config config/config.yaml --model yolov8 --epochs 100

# Mask R-CNN
python train.py --config config/config.yaml --model maskrcnn --epochs 50

# Hybrid model
python train.py --config config/config.yaml --model hybrid --epochs 100
```
**Output**: `checkpoints/best_model.pth`

### 🔍 Inference (Prediction)
```powershell
# Single image
python inference.py --model checkpoints/best_model.pth --source image.jpg

# Folder of images
python inference.py --model checkpoints/best_model.pth --source test_images/

# Video
python inference.py --model checkpoints/best_model.pth --source video.mp4

# Webcam
python inference.py --model checkpoints/best_model.pth --source 0
```

### 📈 Model Evaluation
```powershell
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml
```

### 🎨 Demo Examples
```powershell
python demo.py
```

### 🌐 Web Application
```powershell
streamlit run app.py
```

---

## File Types and Their Purpose

| File Type | Purpose | Can Run Directly? |
|-----------|---------|-------------------|
| `train.py` | Training script | ✅ YES |
| `inference.py` | Prediction script | ✅ YES |
| `evaluate.py` | Evaluation script | ✅ YES |
| `demo.py` | Demo examples | ✅ YES |
| `app.py` | Web app | ✅ YES (with streamlit) |
| `analyze_dataset.py` | Dataset analysis | ✅ YES |
| `src/models/*.py` | Model modules | ❌ NO - Import only |
| `src/dataset/*.py` | Dataset modules | ❌ NO - Import only |
| `src/utils/*.py` | Utility modules | ❌ NO - Import only |

---

## Quick Start Checklist

- [x] ✅ Python 3.8+ installed
- [x] ✅ Virtual environment activated (`damage\Scripts\activate`)
- [x] ✅ Dependencies installed (`pip install -r requirements.txt`)
- [x] ✅ prettytable installed
- [x] ✅ Dataset in `CarDD_release/` folder
- [x] ✅ Dataset analysis completed (`python analyze_dataset.py`)
- [ ] ⏳ Model training (next step: `python train.py`)
- [ ] ⏳ Inference testing (after training)
- [ ] ⏳ Web app deployment (`streamlit run app.py`)

---

## Next Steps

### 1. Start Training (Choose One)

**Fast Training (~2-3 hours GPU)**:
```powershell
python train.py --config config/config.yaml --model yolov8 --backbone yolov8n --epochs 50
```

**Balanced Training (~3-4 hours GPU)**:
```powershell
python train.py --config config/config.yaml --model yolov8 --backbone yolov8s --epochs 100
```

**High Accuracy (~8-12 hours GPU)**:
```powershell
python train.py --config config/config.yaml --model hybrid --epochs 100
```

### 2. Monitor Training

Open new terminal:
```powershell
tensorboard --logdir outputs/
```
Then open: http://localhost:6006

### 3. Test Inference

After training completes:
```powershell
# Test on a single image from test set
python inference.py --model checkpoints/best_model.pth --source CarDD_release/CarDD_COCO/test2017/0001.jpg --output results/
```

### 4. Launch Web App

```powershell
streamlit run app.py
```

---

## Dataset Statistics (From Analysis)

✅ Successfully analyzed!

**Training Set:**
- Images: 2,816
- Annotations: 6,211
- Classes: 6 (dent, scratch, crack, glass shatter, lamp broken, tire flat)
- Most common: scratch (41.22%), dent (29.08%)

**Validation Set:**
- Images: 810
- Annotations: 1,744

**Test Set:**
- Images: 374
- Annotations: 785

---

## Troubleshooting

### "No such file or directory"
✅ Make sure you're in the project root: `cd e:\car_damage_latest`

### "Module not found"
✅ Install dependencies: `pip install -r requirements.txt`

### "No trained model"
✅ You need to train first: `python train.py --config config/config.yaml`

### "Streamlit command not found"
✅ Install streamlit: `pip install streamlit`

---

**Ready to start training? Run:**
```powershell
python train.py --config config/config.yaml --model yolov8 --backbone yolov8s --epochs 100
```

This will take 3-4 hours on GPU (or 12-16 hours on CPU) and create `checkpoints/best_model.pth`
