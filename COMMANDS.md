# ⚡ Quick Command Reference

## ✅ Training Commands

### Basic Training (Uses config.yaml settings)
```powershell
python train.py --config config/config.yaml
```

### Train with Command-Line Overrides
```powershell
# YOLOv8 small model, 100 epochs
python train.py --config config/config.yaml --model yolov8 --backbone yolov8s --epochs 100

# YOLOv8 nano (fastest), 50 epochs
python train.py --config config/config.yaml --model yolov8 --backbone yolov8n --epochs 50

# YOLOv8 medium, custom batch size and learning rate
python train.py --config config/config.yaml --model yolov8 --backbone yolov8m --epochs 100 --batch-size 8 --lr 0.001

# Mask R-CNN model
python train.py --config config/config.yaml --model maskrcnn --epochs 50

# Hybrid model (YOLO + Segmentation)
python train.py --config config/config.yaml --model hybrid --epochs 100

# Resume training from checkpoint
python train.py --config config/config.yaml --resume checkpoints/last_model.pth
```

### Training Options Reference
| Argument | Options | Description | Example |
|----------|---------|-------------|---------|
| `--config` | path | Config file path | `config/config.yaml` |
| `--model` | yolov8, maskrcnn, hybrid | Model architecture | `--model yolov8` |
| `--backbone` | yolov8n/s/m/l/x | Model size (YOLOv8 only) | `--backbone yolov8s` |
| `--epochs` | number | Training epochs | `--epochs 100` |
| `--batch-size` | number | Batch size | `--batch-size 16` |
| `--lr` | float | Learning rate | `--lr 0.001` |
| `--device` | cpu, cuda, mps | Device to use | `--device cuda` |
| `--resume` | path | Resume from checkpoint | `--resume checkpoints/last.pth` |

---

## ✅ Inference Commands

### Single Image
```powershell
# Basic prediction
python inference.py --model checkpoints/best_model.pth --source image.jpg --output results/

# Custom confidence threshold
python inference.py --model checkpoints/best_model.pth --source image.jpg --output results/ --conf 0.35

# Use specific config file
python inference.py --model checkpoints/best_model.pth --config config/config.yaml --source image.jpg --output results/
```

### Batch Processing (Folder of Images)
```powershell
python inference.py --model checkpoints/best_model.pth --source test_images/ --output results/
```

### Video Processing
```powershell
# Process video file
python inference.py --model checkpoints/best_model.pth --source video.mp4 --output results/

# Show live preview while processing
python inference.py --model checkpoints/best_model.pth --source video.mp4 --output results/ --show
```

### Webcam Real-time Detection
```powershell
# Use default webcam (camera 0)
python inference.py --model checkpoints/best_model.pth --source 0 --output results/

# Use external webcam (camera 1)
python inference.py --model checkpoints/best_model.pth --source 1 --output results/
```

### Inference Options Reference
| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--model` | ✅ YES | Path to model checkpoint | `checkpoints/best_model.pth` |
| `--source` | ✅ YES | Image/folder/video/webcam | `image.jpg` or `0` |
| `--output` | No | Output directory | `results/` (default: outputs) |
| `--config` | No | Config file path | `config/config.yaml` |
| `--conf` | No | Confidence threshold | `0.25` (default) |
| `--device` | No | Device to use | `cuda` (default) |
| `--show` | No | Show preview (video) | Add flag to enable |

---

## ✅ Evaluation Commands

```powershell
# Evaluate on test set
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml

# Custom confidence threshold
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml --conf-threshold 0.35

# Evaluate on specific split
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml --split test
```

---

## ✅ Dataset Analysis

```powershell
# Analyze all splits (train, val, test)
python analyze_dataset.py

# Generates:
# - Console statistics
# - dataset_analysis_train.png
# - dataset_analysis_val.png
# - dataset_analysis_test.png
```

---

## ✅ Demo Examples

```powershell
# Run all demo examples
python demo.py

# Demonstrates:
# 1. Single image detection
# 2. Batch processing
# 3. Video processing
# 4. Custom visualization
# 5. Model comparison
# 6. Ensemble prediction
```

---

## ✅ Web Application

```powershell
# Start Streamlit web app
streamlit run app.py

# Access at: http://localhost:8501

# Run on specific port
streamlit run app.py --server.port 8080

# Allow external access
streamlit run app.py --server.address 0.0.0.0
```

---

## ✅ Model Export

```powershell
# Export to ONNX
python export.py --model checkpoints/best_model.pth --format onnx

# Export to multiple formats
python export.py --model checkpoints/best_model.pth --format onnx,torchscript,tflite

# Specify output directory
python export.py --model checkpoints/best_model.pth --format onnx --output exports/
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Start (Fastest)
```powershell
# 1. Analyze dataset
python analyze_dataset.py

# 2. Train smallest/fastest model
python train.py --config config/config.yaml --model yolov8 --backbone yolov8n --epochs 50

# 3. Test on single image
python inference.py --model checkpoints/best_model.pth --source test.jpg --output results/

# 4. Launch web app
streamlit run app.py
```

### Workflow 2: Best Accuracy
```powershell
# 1. Analyze dataset
python analyze_dataset.py

# 2. Train larger model with more epochs
python train.py --config config/config.yaml --model yolov8 --backbone yolov8x --epochs 200

# 3. Evaluate performance
python evaluate.py --model checkpoints/best_model.pth --config config/config.yaml

# 4. Process test images
python inference.py --model checkpoints/best_model.pth --source test_images/ --output results/
```

### Workflow 3: Production Deployment
```powershell
# 1. Train optimized model
python train.py --config config/config.yaml --model yolov8 --backbone yolov8m --epochs 150

# 2. Export to ONNX for deployment
python export.py --model checkpoints/best_model.pth --format onnx

# 3. Deploy web app
streamlit run app.py --server.address 0.0.0.0
```

---

## 📊 Model Comparison

### YOLOv8 Model Sizes

| Model | Parameters | Speed (GPU) | Accuracy | Use Case |
|-------|-----------|------------|----------|----------|
| YOLOv8n | 3.2M | ~5ms | Good | Real-time, edge devices |
| YOLOv8s | 11.2M | ~10ms | Better | Balanced speed/accuracy |
| YOLOv8m | 25.9M | ~15ms | High | Production quality |
| YOLOv8l | 43.7M | ~20ms | Very High | High accuracy needed |
| YOLOv8x | 68.2M | ~30ms | Best | Maximum accuracy |

### Recommended Settings

**For Development/Testing:**
```powershell
python train.py --config config/config.yaml --model yolov8 --backbone yolov8n --epochs 50
```

**For Production (Balanced):**
```powershell
python train.py --config config/config.yaml --model yolov8 --backbone yolov8s --epochs 100
```

**For Maximum Accuracy:**
```powershell
python train.py --config config/config.yaml --model yolov8 --backbone yolov8x --epochs 200
```

---

## 🔧 Troubleshooting

### Training Issues

**Out of Memory:**
```powershell
# Reduce batch size
python train.py --config config/config.yaml --model yolov8 --backbone yolov8n --batch-size 4
```

**Training Too Slow:**
```powershell
# Use smaller model and fewer epochs
python train.py --config config/config.yaml --model yolov8 --backbone yolov8n --epochs 30
```

### Inference Issues

**No Detections:**
```powershell
# Lower confidence threshold
python inference.py --model checkpoints/best_model.pth --source image.jpg --conf 0.15
```

**Slow Inference:**
```powershell
# Use CPU if GPU is slow
python inference.py --model checkpoints/best_model.pth --source image.jpg --device cpu
```

---

## 💡 Pro Tips

1. **Monitor Training**: Use TensorBoard
   ```powershell
   tensorboard --logdir outputs/
   ```

2. **Resume if Interrupted**: Always use --resume
   ```powershell
   python train.py --config config/config.yaml --resume checkpoints/last_model.pth
   ```

3. **Test Multiple Thresholds**: Find optimal confidence
   ```powershell
   python inference.py --model checkpoints/best_model.pth --source test.jpg --conf 0.2
   python inference.py --model checkpoints/best_model.pth --source test.jpg --conf 0.3
   python inference.py --model checkpoints/best_model.pth --source test.jpg --conf 0.4
   ```

4. **Batch Process Efficiently**: Use folder input
   ```powershell
   python inference.py --model checkpoints/best_model.pth --source test_folder/ --output results/
   ```

---

## ✅ Current Status Checklist

- [x] ✅ Environment setup complete
- [x] ✅ Dependencies installed
- [x] ✅ Dataset analyzed
- [ ] ⏳ **Next: Start training** 👇

```powershell
python train.py --config config/config.yaml --model yolov8 --backbone yolov8s --epochs 100
```

This will take approximately **3-4 hours on GPU** or **12-16 hours on CPU**.
