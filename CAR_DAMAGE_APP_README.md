# 🚗 CarDD - AI Car Damage Detection Streamlit App

## Overview

Professional web application for AI-powered vehicle damage assessment using the trained YOLOv8 model from the CarDD dataset.

## 🌟 Features

### Core Functionality
- ✅ **Real-time Damage Detection** - Upload images and get instant damage analysis
- ✅ **6 Damage Types** - Detects dents, scratches, cracks, glass shatter, lamp broken, tire flat
- ✅ **Visual Annotations** - Color-coded bounding boxes with confidence scores
- ✅ **Batch Processing** - Process multiple images simultaneously
- ✅ **Interactive Dashboard** - Analytics and charts for detection insights
- ✅ **Severity Assessment** - Automatic severity classification (Minor/Moderate/Severe)

### Advanced Features
- 📊 **Detection Analytics** - Interactive charts and statistics
- 💾 **Export Results** - Download annotated images and JSON reports
- ⚙️ **Configurable Settings** - Adjust confidence threshold and display options
- 🎯 **GPU Acceleration** - Automatic GPU detection for faster inference
- 📱 **Responsive Design** - Professional UI with custom styling

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Trained YOLOv8 model (from the notebook)
- GPU (optional, but recommended for faster inference)

### Step 1: Install Dependencies

```bash
# Activate your virtual environment
E:\car_damage_latest\damage\Scripts\Activate.ps1

# Install required packages
pip install streamlit
pip install ultralytics
pip install opencv-python
pip install plotly
pip install pandas
pip install pillow
```

### Step 2: Verify Model Path

Make sure your trained model is at:
```
runs/detect/outputs/training/car_damage_colab/weights/best.pt
```

Or update the path in the app's sidebar.

## 🎯 Usage

### Starting the App

```bash
# Navigate to project directory
cd E:\car_damage_latest

# Run the Streamlit app
streamlit run car_damage_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the App

#### 1. **Single Image Detection**
   - Click "📸 Single Image Detection" tab
   - Upload an image (JPG, PNG, BMP)
   - Click "🔍 Detect Damage"
   - View results with:
     - Annotated image with bounding boxes
     - Severity assessment
     - Detailed damage breakdown
     - Confidence scores
   - Download results (image + JSON report)

#### 2. **Batch Processing**
   - Click "📁 Batch Processing" tab
   - Upload multiple images
   - Click "🚀 Process All Images"
   - View batch statistics
   - Download comprehensive batch report

#### 3. **Analytics Dashboard**
   - View detection charts
   - Confidence distribution
   - Detection statistics table
   - Export analytics data

#### 4. **User Guide**
   - Complete documentation
   - Tips for best results
   - Troubleshooting guide

## ⚙️ Configuration

### Sidebar Settings

**Model Settings:**
- **Model Path**: Path to trained .pt model file
- **Confidence Threshold**: 0.0 - 1.0 (default: 0.25)

**Detection Options:**
- Show Labels (checkbox)
- Show Confidence (checkbox)
- Show Bounding Boxes (checkbox)

**Image Settings:**
- Max Image Size: 320-1280px (default: 640)

## 📊 Damage Classes

| Class ID | Damage Type | Color Code |
|----------|------------|------------|
| 0 | Dent | Blue |
| 1 | Scratch | Green |
| 2 | Crack | Red |
| 3 | Glass Shatter | Cyan |
| 4 | Lamp Broken | Magenta |
| 5 | Tire Flat | Yellow |

## 🎨 Output Format

### Detection Report (JSON)
```json
{
  "timestamp": "2026-02-18 10:30:00",
  "image": "car_image.jpg",
  "total_damages": 3,
  "average_confidence": 0.85,
  "severity_assessment": "Moderate",
  "damage_breakdown": {
    "dent": 2,
    "scratch": 1
  },
  "detections": [
    {
      "class": "dent",
      "confidence": 0.92,
      "bbox": [100, 150, 300, 400],
      "area": 50000
    }
  ]
}
```

## 💡 Tips for Best Results

1. **Image Quality**
   - Use clear, well-lit images
   - Avoid blurry or low-resolution photos
   - Ensure damaged area is clearly visible

2. **Camera Angle**
   - Take photos from multiple angles
   - Get close-up shots of damages
   - Avoid excessive shadows

3. **Confidence Threshold**
   - Lower: More detections (may include false positives)
   - Higher: Fewer detections (only high confidence)
   - Recommended: 0.25 - 0.35 for most cases

4. **Batch Processing**
   - Process similar images together
   - Use consistent lighting conditions
   - Limit to 20-30 images per batch for performance

## 🔧 Troubleshooting

### Model Not Loading

**Issue:** Model file not found

**Solution:**
1. Verify model path in sidebar
2. Check if `best.pt` exists at specified location
3. Train model using the notebook first if missing

### No Detections

**Issue:** No damages detected in image

**Solutions:**
- Lower confidence threshold (try 0.15-0.20)
- Ensure image shows actual damage
- Check image quality and lighting
- Try different angles

### Slow Performance

**Issue:** Detection takes too long

**Solutions:**
- Enable GPU/CUDA for faster inference
- Reduce max image size in settings
- Close other GPU-intensive applications
- Use batch processing for multiple images

### Import Errors

**Issue:** Module not found errors

**Solution:**
```bash
pip install streamlit ultralytics opencv-python plotly pandas pillow
```

## 📈 System Requirements

### Minimum Requirements
- **CPU:** Intel i5 or equivalent
- **RAM:** 8GB
- **Storage:** 2GB free space
- **OS:** Windows 10/11, Linux, macOS

### Recommended Requirements
- **CPU:** Intel i7 or equivalent
- **RAM:** 16GB
- **GPU:** NVIDIA GPU with CUDA support (RTX 2060+)
- **Storage:** 5GB free space

## 🔄 Updates & Maintenance

### Model Updates
To use a new trained model:
1. Train model using the notebook
2. Copy `best.pt` to desired location
3. Update model path in app sidebar
4. Restart app

### App Updates
Check the notebook for latest app version and features.

## 📝 Known Limitations

1. **Model Size:** YOLOv8m requires ~50MB storage
2. **GPU Memory:** Batch processing limited by available VRAM
3. **Image Size:** Very large images (>4K) may be slow without GPU
4. **Browser Support:** Best in Chrome, Firefox, Edge

## 🆘 Support

### Common Issues

**Q: App won't start**  
A: Check if port 8501 is available, or specify a different port:
```bash
streamlit run car_damage_app.py --server.port 8502
```

**Q: GPU not detected**  
A: Ensure PyTorch with CUDA is installed correctly (see notebook for GPU setup)

**Q: Out of memory errors**  
A: Reduce batch size or max image size in settings

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Average Detection Time (CPU) | 2-5 seconds |
| Average Detection Time (GPU) | 0.2-0.5 seconds |
| Supported Image Formats | JPG, PNG, BMP |
| Max Batch Size | 50+ images |
| Model Size | ~50MB |

## 🔗 Related Files

- **Training Notebook:** `car_damage_colab.ipynb`
- **Model Weights:** `runs/detect/outputs/training/car_damage_colab/weights/best.pt`
- **Dataset:** `CarDD_release/`

## 📄 License

This app is built for educational and research purposes using the CarDD dataset.

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **CarDD Dataset** for training data
- **Streamlit** for web framework
- **PyTorch** for deep learning

---

## Quick Start Example

```bash
# 1. Activate environment
E:\car_damage_latest\damage\Scripts\Activate.ps1

# 2. Install dependencies (if not already installed)
pip install streamlit ultralytics opencv-python plotly

# 3. Run the app
streamlit run car_damage_app.py

# 4. Open browser to http://localhost:8501

# 5. Upload a car image and click "Detect Damage"
```

---

**Version:** 1.0.0  
**Last Updated:** February 18, 2026  
**Author:** AI Assistant

**Happy Damage Detection! 🚗✨**
