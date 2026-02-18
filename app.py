"""
Streamlit Web Application for Car Damage Detection
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import time
import json
import tempfile
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.yolo_model import create_yolo_model
from src.models.resnet50_classifier import create_resnet50_classifier, CLASS_NAMES as RESNET_CLASS_NAMES
from src.utils.visualization import visualize_predictions, CLASS_NAMES, CLASS_COLORS
from src.utils.logger import setup_logger


# Page configuration
st.set_page_config(
    page_title="CarDD - Car Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .damage-card {
        background-color: #ffffff;
        border-left: 4px solid #FF4B4B;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str, config_path: str = None, model_type: str = 'auto'):
    """Load model with caching"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if it's ResNet50 pretrained model
        if model_type == 'resnet50' or model_path.endswith('resnet50-19c8e357.pth'):
            st.info("🔍 Loading ResNet50 pretrained model for damage classification...")
            model = create_resnet50_classifier(
                num_classes=6,
                pretrained_path=model_path if Path(model_path).exists() else None,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            config = {
                'model': {'architecture': 'resnet50'},
                'dataset': {'num_classes': 6}
            }
            return model, device, config
        
        # Load config for other models
        if not config_path:
            config_path = 'config/config.yaml'
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model_arch = config['model']['architecture']
        num_classes = config['dataset']['num_classes']
        
        if model_arch == 'yolov8':
            from src.models.yolo_model import create_yolo_model
            model = create_yolo_model(
                model_size=config['model']['backbone'].replace('yolov8', ''),
                num_classes=num_classes,
                pretrained=False
            )
        else:
            st.error(f"Model architecture {model_arch} not supported in web app yet")
            return None, None, None
        
        # Load checkpoint if exists
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.model.load(model_path)
        
        model.model.to(device)
        model.model.eval()
        
        return model, device, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None


def process_image(image: np.ndarray, model, device, conf_threshold: float, model_type: str = 'yolov8'):
    """Process image and return predictions"""
    try:
        # Demo mode - create synthetic detections
        if model_type == 'demo':
            start_time = time.time()
            h, w = image.shape[:2]
            
            # Create synthetic detections for demo
            import random
            random.seed(42)  # Consistent results
            
            num_detections = random.randint(1, 4)
            detections = []
            
            for i in range(num_detections):
                # Random damage type
                damage_idx = random.randint(0, 5)
                
                # Random bounding box
                x1 = random.randint(int(w * 0.1), int(w * 0.6))
                y1 = random.randint(int(h * 0.1), int(h * 0.6))
                box_w = random.randint(int(w * 0.15), int(w * 0.35))
                box_h = random.randint(int(h * 0.15), int(h * 0.35))
                x2 = min(x1 + box_w, w)
                y2 = min(y1 + box_h, h)
                
                # Random confidence
                confidence = random.uniform(0.6, 0.95)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': CLASS_NAMES[damage_idx],
                    'confidence': confidence
                })
            
            # Convert to YOLO-like format
            class PseudoPred:
                def __init__(self, detections):
                    boxes = []
                    labels = []
                    scores = []
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        boxes.append([x1, y1, x2, y2])
                        class_idx = CLASS_NAMES.index(det['class'])
                        labels.append(class_idx)
                        scores.append(det['confidence'])
                    
                    class Boxes:
                        def __init__(self, boxes, labels, scores):
                            self.xyxy = torch.tensor(boxes)
                            self.cls = torch.tensor(labels)
                            self.conf = torch.tensor(scores)
                    
                    self.boxes = Boxes(boxes, labels, scores)
            
            predictions = [PseudoPred(detections)]
            inference_time = time.time() - start_time + random.uniform(0.01, 0.05)
            
            return predictions, inference_time
        
        # Check if ResNet50 classifier
        if hasattr(model, 'predict_with_sliding_window'):
            # ResNet50 classifier - use sliding window detection
            start_time = time.time()
            pil_image = Image.fromarray(image)
            detections = model.predict_with_sliding_window(
                pil_image,
                window_size=224,
                stride=100,
                conf_threshold=conf_threshold
            )
            inference_time = time.time() - start_time
            
            # Convert to YOLO-like format for visualization
            if detections:
                # Create pseudo-predictions object
                class PseudoPred:
                    def __init__(self, detections):
                        boxes = []
                        labels = []
                        scores = []
                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            boxes.append([x1, y1, x2, y2])
                            # Map class name to index
                            class_idx = RESNET_CLASS_NAMES.index(det['class']) if det['class'] in RESNET_CLASS_NAMES else 0
                            labels.append(class_idx)
                            scores.append(det['confidence'])
                        
                        class Boxes:
                            def __init__(self, boxes, labels, scores):
                                self.xyxy = torch.tensor(boxes)
                                self.cls = torch.tensor(labels)
                                self.conf = torch.tensor(scores)
                        
                        self.boxes = Boxes(boxes, labels, scores)
                
                predictions = [PseudoPred(detections)]
            else:
                predictions = []
            
            return predictions, inference_time
        
        # Regular YOLO model
        # Prepare image
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predictions = model.predict(img_tensor, conf=conf_threshold)
        inference_time = time.time() - start_time
        
        return predictions, inference_time
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, 0



def create_damage_summary(predictions):
    """Create summary statistics from predictions"""
    if not predictions or len(predictions) == 0:
        return None
    
    pred = predictions[0] if isinstance(predictions, list) else predictions
    
    if len(pred.boxes) == 0:
        return None
    
    boxes = pred.boxes.xyxy.cpu().numpy()
    labels = pred.boxes.cls.cpu().numpy().astype(int)
    scores = pred.boxes.conf.cpu().numpy()
    
    summary = {
        'total_damages': len(boxes),
        'avg_confidence': float(np.mean(scores)),
        'damages_by_type': {},
        'details': []
    }
    
    for box, label, score in zip(boxes, labels, scores):
        damage_type = CLASS_NAMES[label]
        if damage_type not in summary['damages_by_type']:
            summary['damages_by_type'][damage_type] = 0
        summary['damages_by_type'][damage_type] += 1
        
        summary['details'].append({
            'type': damage_type,
            'confidence': float(score),
            'bbox': box.tolist()
        })
    
    return summary


def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 CarDD - Car Damage Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Vehicle Damage Assessment System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/car-damage.png", width=100)
        st.markdown("### ⚙️ Settings")
        
        # Model type selection
        model_type = st.selectbox(
            "Model Type",
            options=['Demo Mode (No Training)', 'ResNet50 Pretrained', 'YOLOv8 (Trained)', 'Auto Detect'],
            index=0,
            help="Select model type to use"
        )
        
        # Model selection
        if model_type == 'Demo Mode (No Training)':
            st.info("🎨 Demo Mode: Shows synthetic detections for testing the UI")
            model_path = None
            config_path = None
        elif model_type == 'ResNet50 Pretrained':
            st.warning("⚠️ ResNet50 pretrained won't detect damage without fine-tuning. Use 'Demo Mode' or train YOLOv8.")
            model_path = st.text_input(
                "Model Path",
                value="resnet50-19c8e357.pth",
                help="Path to ResNet50 pretrained weights (or leave default to download)"
            )
            config_path = None
        else:
            model_path = st.text_input(
                "Model Path",
                value="checkpoints/best_model.pth",
                help="Path to your trained model checkpoint"
            )
            config_path = st.text_input(
                "Config Path",
                value="config/config.yaml",
                help="Path to configuration file"
            )

        # Load model button
        if st.button("🔄 Load Model"):
            with st.spinner("Loading model..."):
                if model_type == 'ResNet50 Pretrained':
                    st.session_state.model_data = load_model(model_path, None, 'resnet50')
                    st.session_state.current_model_type = 'resnet50'
                else:
                    st.session_state.model_data = load_model(model_path, config_path, 'auto')
                    st.session_state.current_model_type = 'yolov8'
                
                if st.session_state.model_data[0] is not None:
                    st.success("✅ Model loaded successfully!")
                    if model_type == 'ResNet50 Pretrained':
                        st.info("ℹ️ ResNet50 uses sliding window detection. This may be slower but works without training.")
                else:
                    st.error("❌ Failed to load model")
        
        st.markdown("---")
        
        # Detection settings
        st.markdown("### 🎯 Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        show_labels = st.checkbox("Show Labels", value=True)
        show_conf = st.checkbox("Show Confidence", value=True)
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.info("""
        **CarDD** detects 6 types of car damage:
        - 🔴 Dent
        - 🟢 Scratch
        - 🔵 Crack
        - 🟡 Glass Shatter
        - 🟣 Lamp Broken
        - 🔵 Tire Flat
        
        Upload an image or use webcam to detect damage.
        """)
        
        # Statistics
        if 'detection_count' not in st.session_state:
            st.session_state.detection_count = 0
        
        st.markdown("---")
        st.metric("Total Detections", st.session_state.detection_count)
    
    # Main content
    tabs = st.tabs(["📸 Image Upload", "📹 Webcam", "🎞️ Video Upload", "📊 Analytics", "ℹ️ Help"])
    
    # Tab 1: Image Upload
    with tabs[0]:
        st.markdown("### Upload an Image")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a car image to detect damage"
            )
            
            if uploaded_file is not None:
                # Display original image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                st.image(image, caption="Original Image", use_column_width=True)
                
                # Process button
                if st.button("🔍 Detect Damage", key="detect_btn"):
                    if 'model_data' not in st.session_state or st.session_state.model_data[0] is None:
                        st.warning("⚠️ Please load the model first from the sidebar!")
                    else:
                        model, device, config = st.session_state.model_data
                        
                        with st.spinner("🔍 Analyzing image..."):
                            # Get model type
                            current_model_type = st.session_state.get('current_model_type', 'yolov8')
                            
                            # Process image
                            predictions, inference_time = process_image(
                                image_np, model, device, conf_threshold, current_model_type
                            )
                            
                            if predictions:
                                st.session_state.current_predictions = predictions
                                st.session_state.current_image = image_np
                                st.session_state.inference_time = inference_time
                                st.session_state.detection_count += 1
        
        with col2:
            if 'current_predictions' in st.session_state:
                # Display annotated image
                pred = st.session_state.current_predictions[0]
                
                # Convert predictions to dict format
                pred_dict = {
                    'boxes': pred.boxes.xyxy,
                    'labels': pred.boxes.cls.long(),
                    'scores': pred.boxes.conf
                }
                
                annotated = visualize_predictions(
                    st.session_state.current_image,
                    pred_dict,
                    conf_threshold=conf_threshold,
                    show_labels=show_labels,
                    show_conf=show_conf
                )
                
                st.image(annotated, caption="Detection Result", use_column_width=True)
                
                # Show metrics
                st.markdown("### 📊 Detection Results")
                
                col_a, col_b, col_c = st.columns(3)
                
                summary = create_damage_summary(st.session_state.current_predictions)
                
                if summary:
                    with col_a:
                        st.metric("🔍 Damages Found", summary['total_damages'])
                    with col_b:
                        st.metric("⚡ Inference Time", f"{st.session_state.inference_time*1000:.1f} ms")
                    with col_c:
                        st.metric("🎯 Avg Confidence", f"{summary['avg_confidence']:.1%}")
                    
                    # Damage breakdown
                    st.markdown("### 📋 Damage Breakdown")
                    for damage_type, count in summary['damages_by_type'].items():
                        st.markdown(f"""
                        <div class="damage-card">
                            <strong>{damage_type.upper()}</strong>: {count} instance(s)
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed results
                    with st.expander("📝 Detailed Results"):
                        df = pd.DataFrame(summary['details'])
                        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(df[['type', 'confidence']], use_container_width=True)
                    
                    # Download results
                    st.markdown("### 💾 Download")
                    
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        # Download annotated image
                        annotated_pil = Image.fromarray(annotated)
                        buf = BytesIO()
                        annotated_pil.save(buf, format='PNG')
                        st.download_button(
                            label="📥 Download Image",
                            data=buf.getvalue(),
                            file_name="damage_detection.png",
                            mime="image/png"
                        )
                    
                    with col_d2:
                        # Download JSON results
                        json_str = json.dumps(summary, indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name="damage_report.json",
                            mime="application/json"
                        )
                else:
                    st.success("✅ No damage detected!")
    
    # Tab 2: Webcam
    with tabs[1]:
        st.markdown("### 📹 Webcam Detection")
        st.info("🚧 Webcam feature requires streamlit-webrtc. Install with: `pip install streamlit-webrtc`")
        st.markdown("""
        To enable webcam detection:
        1. Install streamlit-webrtc
        2. Allow camera access in your browser
        3. Click 'Start' to begin detection
        """)
    
    # Tab 3: Video Upload
    with tabs[2]:
        st.markdown("### 🎞️ Video Upload")
        
        video_file = st.file_uploader(
            "Upload a video",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video for damage detection"
        )
        
        if video_file is not None:
            # Save uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            
            st.video(tfile.name)
            
            if st.button("🎬 Process Video"):
                if 'model_data' not in st.session_state or st.session_state.model_data[0] is None:
                    st.warning("⚠️ Please load the model first!")
                else:
                    st.info("🚧 Video processing feature coming soon!")
                    st.markdown("""
                    For now, you can process videos using the CLI:
                    ```bash
                    python inference.py --model checkpoints/best_model.pth --source video.mp4 --output outputs/
                    ```
                    """)
    
    # Tab 4: Analytics
    with tabs[3]:
        st.markdown("### 📊 Analytics Dashboard")
        
        if 'current_predictions' in st.session_state:
            summary = create_damage_summary(st.session_state.current_predictions)
            
            if summary and summary['damages_by_type']:
                # Damage distribution pie chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Damage Type Distribution")
                    fig = px.pie(
                        values=list(summary['damages_by_type'].values()),
                        names=list(summary['damages_by_type'].keys()),
                        title="Damage Types",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Confidence Scores")
                    confidences = [d['confidence'] for d in summary['details']]
                    types = [d['type'] for d in summary['details']]
                    
                    fig = px.bar(
                        x=types,
                        y=confidences,
                        title="Detection Confidence by Type",
                        labels={'x': 'Damage Type', 'y': 'Confidence'},
                        color=confidences,
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics table
                st.markdown("#### Summary Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Total Damages', 'Unique Types', 'Avg Confidence', 'Max Confidence', 'Min Confidence'],
                    'Value': [
                        summary['total_damages'],
                        len(summary['damages_by_type']),
                        f"{summary['avg_confidence']:.2%}",
                        f"{max(confidences):.2%}",
                        f"{min(confidences):.2%}"
                    ]
                })
                st.table(stats_df)
        else:
            st.info("👆 Upload and process an image to see analytics")
    
    # Tab 5: Help
    with tabs[4]:
        st.markdown("### 📖 User Guide")
        
        st.markdown("""
        #### How to Use
        
        1. **Load Model**
           - Enter model path in sidebar (default: `checkpoints/best_model.pth`)
           - Click "Load Model" button
           - Wait for success message
        
        2. **Upload Image**
           - Go to "Image Upload" tab
           - Click "Browse files" and select a car image
           - Click "Detect Damage" button
           - View results on the right side
        
        3. **Adjust Settings**
           - Use confidence threshold slider to filter detections
           - Toggle labels and confidence display
        
        4. **Download Results**
           - Download annotated image as PNG
           - Download detection report as JSON
        
        #### Supported Damage Types
        
        | Icon | Type | Description |
        |------|------|-------------|
        | 🔴 | Dent | Body dents and deformations |
        | 🟢 | Scratch | Surface scratches and paint damage |
        | 🔵 | Crack | Body cracks and fractures |
        | 🟡 | Glass Shatter | Broken or shattered glass |
        | 🟣 | Lamp Broken | Damaged headlights or taillights |
        | 🔵 | Tire Flat | Flat or damaged tires |
        
        #### Tips for Best Results
        
        - ✅ Use clear, well-lit images
        - ✅ Ensure damage is visible
        - ✅ Avoid heavy shadows
        - ✅ Use high-resolution images
        - ❌ Avoid blurry images
        - ❌ Avoid extreme angles
        
        #### Troubleshooting
        
        **Model won't load?**
        - Check if model file exists
        - Verify config.yaml path
        - Ensure PyTorch is installed
        
        **No detections?**
        - Lower confidence threshold
        - Check image quality
        - Ensure damage is visible
        
        **Slow inference?**
        - Use smaller model (YOLOv8n)
        - Reduce image resolution
        - Enable GPU if available
        
        #### System Requirements
        
        - Python 3.8+
        - PyTorch 2.0+
        - 4GB+ RAM
        - GPU recommended (optional)
        
        #### Contact & Support
        
        - 📧 Report issues on GitHub
        - 📚 Check documentation
        - 💬 Join community discussions
        """)
        
        # System info
        with st.expander("🖥️ System Information"):
            st.code(f"""
Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}
PyTorch Version: {torch.__version__}
CUDA Available: {torch.cuda.is_available()}
""")


if __name__ == "__main__":
    # Initialize session state
    if 'detection_count' not in st.session_state:
        st.session_state.detection_count = 0
    
    main()
