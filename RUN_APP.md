# 🚀 Running the CarDD Streamlit App

## Quick Start

### 1. Install Streamlit Dependencies

```bash
pip install streamlit plotly streamlit-webrtc
```

Or update your requirements:

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Features

### 📸 Image Upload
- Upload car images (JPG, PNG)
- Real-time damage detection
- Visualized results with bounding boxes
- Confidence scores and labels
- Download annotated images
- Export detection reports (JSON)

### 📹 Webcam (Coming Soon)
- Real-time webcam detection
- Live damage assessment
- Frame-by-frame analysis

### 🎞️ Video Upload
- Upload video files (MP4, AVI, MOV)
- Process entire videos
- Frame-by-frame damage detection

### 📊 Analytics Dashboard
- Damage type distribution (pie chart)
- Confidence score analysis (bar chart)
- Summary statistics
- Detailed metrics

### ⚙️ Settings
- Adjustable confidence threshold
- Toggle labels and confidence display
- Model selection
- Config management

## Usage Guide

### Step 1: Load Model

1. Open sidebar (left panel)
2. Enter model path: `checkpoints/best_model.pth`
3. Enter config path: `config/config.yaml`
4. Click "🔄 Load Model"
5. Wait for success message

### Step 2: Upload Image

1. Go to "📸 Image Upload" tab
2. Click "Browse files"
3. Select a car image with damage
4. Click "🔍 Detect Damage"

### Step 3: View Results

- **Left side**: Original image
- **Right side**: Annotated image with detections
- **Metrics**: Total damages, inference time, confidence
- **Breakdown**: Damage types and counts
- **Details**: Expandable detailed results table

### Step 4: Download Results

- **📥 Download Image**: Get annotated PNG
- **📥 Download JSON**: Get detection report

## Advanced Features

### Custom Confidence Threshold

Adjust the slider in sidebar (0.0 to 1.0):
- **Higher** (0.5-0.9): Fewer, more confident detections
- **Lower** (0.1-0.4): More detections, may include false positives
- **Recommended**: 0.25-0.35

### Analytics Dashboard

View comprehensive analytics:
- Damage distribution pie chart
- Confidence scores bar chart
- Summary statistics table
- Per-detection details

## Configuration

### Model Settings

Edit in sidebar:
```
Model Path: checkpoints/best_model.pth
Config Path: config/config.yaml
```

### Display Options

- ✅ Show Labels: Display damage type names
- ✅ Show Confidence: Display confidence scores

## Troubleshooting

### App won't start?

```bash
# Check streamlit installation
streamlit --version

# Reinstall if needed
pip install --upgrade streamlit
```

### Model not loading?

- Verify model file exists: `checkpoints/best_model.pth`
- Check config file: `config/config.yaml`
- Ensure PyTorch is installed: `pip install torch torchvision`

### Slow performance?

- Use smaller model (YOLOv8n instead of YOLOv8x)
- Reduce image resolution
- Enable GPU if available

### No detections showing?

- Lower confidence threshold
- Check image quality
- Ensure damage is clearly visible

## Keyboard Shortcuts

- `Ctrl + R`: Refresh page
- `Ctrl + Shift + R`: Clear cache and refresh
- `Ctrl + K`: Focus search

## Tips for Best Results

### Image Quality
- ✅ High resolution (800x600+)
- ✅ Good lighting
- ✅ Clear focus on damage
- ❌ Avoid blur
- ❌ Avoid heavy shadows

### Damage Visibility
- Ensure damage is in frame
- Capture damage from multiple angles
- Get close-up shots for small damages

### Batch Processing
For multiple images, use CLI:
```bash
python inference.py --model checkpoints/best_model.pth --source images_dir/ --output results/
```

## Deployment

### Local Network Access

```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices: `http://YOUR_IP:8501`

### Production Deployment

Deploy to Streamlit Cloud:
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

### Docker Deployment

```dockerfile
FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Performance Optimization

### Cache Model Loading

Models are automatically cached using `@st.cache_resource`

### Clear Cache

```python
# In app sidebar
st.cache_resource.clear()
```

Or press `C` in browser

## Customization

### Change Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Add Custom Metrics

Edit `app.py` and add to analytics tab:

```python
st.metric("Custom Metric", value)
```

## Support

- 📖 Documentation: README.md
- 🐛 Report Issues: GitHub Issues
- 💬 Community: Discussions
- 📧 Contact: your-email@example.com

## Next Steps

After using the app:
1. Export detection reports
2. Analyze patterns in damages
3. Generate statistics
4. Create custom reports
5. Integrate with your workflow

---

**Enjoy using CarDD Web App! 🚗💥🔍**
