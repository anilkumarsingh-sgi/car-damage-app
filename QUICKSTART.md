# Quick Start Guide for CarDD Model

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Analyze Your Dataset

```bash
python analyze_dataset.py
```

This will generate statistics and visualizations for all dataset splits.

### 3. Train Your First Model

#### Option A: YOLOv8 (Recommended for Beginners)

```bash
python train.py --config config/config.yaml
```

#### Option B: Quick Test with Pretrained YOLOv8

```bash
from ultralytics import YOLO

# Load pretrained YOLO
model = YOLO('yolov8x-seg.pt')

# Train on CarDD
model.train(
    data='cardd.yaml',  # Create this from config
    epochs=100,
    batch=16,
    imgsz=640
)
```

### 4. Run Inference

```bash
# Single image
python inference.py --model checkpoints/best_model.pth --source test_image.jpg --output results/

# Webcam
python inference.py --model checkpoints/best_model.pth --source webcam
```

## 📊 Expected Results

After training for 100 epochs:
- **YOLOv8x**: ~79% mAP
- **Training time**: ~8-12 hours on RTX 3090
- **Inference speed**: 45 FPS on GPU

## 🎯 Common Tasks

### Visualize Predictions

```python
from src.utils.visualization import visualize_predictions
import cv2

image = cv2.imread('car_damage.jpg')
predictions = model.predict(image)
result = visualize_predictions(image, predictions)
cv2.imwrite('result.jpg', result)
```

### Export Model

```bash
# To ONNX
python export.py --model checkpoints/best_model.pth --format onnx

# To TorchScript  
python export.py --model checkpoints/best_model.pth --format torchscript
```

### Evaluate Model

```bash
python evaluate.py --model checkpoints/best_model.pth --split test --save
```

## 🔧 Troubleshooting

### CUDA Out of Memory

```yaml
# In config/config.yaml, reduce:
training:
  batch_size: 8  # or 4
  use_amp: true  # Enable mixed precision
```

### Slow Training

```yaml
# Enable optimizations:
training:
  num_workers: 8  # Increase data loading workers
  use_amp: true
```

### Low Accuracy

```yaml
# Increase augmentation:
augmentation:
  train:
    - Increase augmentation probability
    - Add more transforms
```

## 📱 Mobile Deployment

```bash
# Export to TensorFlow Lite
python export.py --model checkpoints/best_model.pth --format tflite

# Export to CoreML (for iOS)
python export.py --model checkpoints/best_model.pth --format coreml
```

## 🌐 Web Demo

Coming soon! Integration with Gradio for easy web interface.

## 💡 Pro Tips

1. **Use smaller model for faster training**: Start with YOLOv8n
2. **Enable WandB**: Set `use_wandb: true` for better monitoring
3. **Use pretrained weights**: Always start with `pretrained: true`
4. **Data augmentation**: Essential for good generalization
5. **Early stopping**: Enabled by default with patience=20

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [CarDD Paper](https://ieeexplore.ieee.org/document/10077382)
- [COCO Format Guide](https://cocodataset.org/#format-data)

---

**Need help?** Open an issue or check the full README.md
