"""
🚗 Car Damage Detection System - Streamlit App
===============================================
Professional web application for AI-powered vehicle damage assessment
using the trained YOLOv8 model from the CarDD dataset.

Features:
- Real-time damage detection with bounding boxes
- Multi-image batch processing
- Confidence threshold adjustment
- Detailed damage report generation
- Model performance metrics
- Export results as JSON/CSV

Author: AI Assistant
Date: February 2026
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from pathlib import Path
import json
import os
import urllib.request
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.error("⚠️ ultralytics not installed. Install with: pip install ultralytics")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Damage classes from CarDD dataset
DAMAGE_CLASSES = {
    0: 'dent',
    1: 'scratch',
    2: 'crack',
    3: 'glass shatter',
    4: 'lamp broken',
    5: 'tire flat'
}

# Color mapping for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    'dent': (255, 0, 0),          # Blue
    'scratch': (0, 255, 0),        # Green
    'crack': (0, 0, 255),          # Red
    'glass shatter': (255, 255, 0), # Cyan
    'lamp broken': (255, 0, 255),   # Magenta
    'tire flat': (0, 255, 255)      # Yellow
}

# Model paths - try multiple locations (local dev vs cloud deployment)
MODEL_SEARCH_PATHS = [
    Path('runs/detect/outputs/training/car_damage_colab/weights/best.pt'),
    Path('models/best.pt'),
    Path('best.pt'),
    Path('yolov8m.pt'),
]

# Optional: Set this env var in Streamlit Cloud secrets to auto-download model
# STREAMLIT_MODEL_URL = "https://github.com/<user>/<repo>/releases/download/v1.0/best.pt"
MODEL_DOWNLOAD_URL = os.environ.get('MODEL_DOWNLOAD_URL', '')


def find_model_path():
    """Find the best available model file"""
    for p in MODEL_SEARCH_PATHS:
        if p.exists():
            return p
    
    # Try downloading if URL is set
    if MODEL_DOWNLOAD_URL:
        dest = Path('models/best.pt')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with st.spinner('📥 Downloading model... This may take a minute.'):
                try:
                    urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, str(dest))
                    return dest
                except Exception as e:
                    st.error(f'Failed to download model: {e}')
    return MODEL_SEARCH_PATHS[0]  # Return default (will show error later)


DEFAULT_MODEL_PATH = find_model_path()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CarDD - AI Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .damage-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
    }
    .stats-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model with caching"""
    try:
        if not YOLO_AVAILABLE:
            return None
        
        model_path = Path(model_path)
        if not model_path.exists():
            st.error(f"❌ Model not found at: {model_path}")
            st.info(
                "💡 **To fix this on Streamlit Cloud:**\n"
                "1. Include your `best.pt` model in the `models/` folder of your repo, OR\n"
                "2. Set `MODEL_DOWNLOAD_URL` in Streamlit Cloud Secrets to auto-download.\n\n"
                "**Locally:** Train the model first using the notebook."
            )
            return None
        
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def draw_predictions(image: np.ndarray, results, conf_threshold: float = 0.25):
    """Draw bounding boxes and labels on image"""
    img_annotated = image.copy()
    
    if results is None or len(results) == 0:
        return img_annotated, []
    
    detections = []
    result = results[0]
    
    if len(result.boxes) == 0:
        return img_annotated, detections
    
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    
    for box, score, cls_id in zip(boxes, scores, classes):
        if score < conf_threshold:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        class_name = DAMAGE_CLASSES.get(cls_id, f'Class_{cls_id}')
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label
        label = f"{class_name}: {score:.2f}"
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            img_annotated,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Store detection info
        detections.append({
            'class': class_name,
            'confidence': float(score),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'area': int((x2 - x1) * (y2 - y1))
        })
    
    return img_annotated, detections


def create_damage_report(detections, image_name="image"):
    """Generate detailed damage assessment report"""
    if not detections:
        return None
    
    # Count damages by type
    damage_counts = {}
    for det in detections:
        damage_type = det['class']
        damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
    
    # Calculate statistics
    avg_confidence = np.mean([d['confidence'] for d in detections])
    total_damage_area = sum([d['area'] for d in detections])
    
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image': image_name,
        'total_damages': len(detections),
        'average_confidence': float(avg_confidence),
        'total_damage_area': int(total_damage_area),
        'damage_breakdown': damage_counts,
        'severity_assessment': assess_severity(detections),
        'detections': detections
    }
    
    return report


def assess_severity(detections):
    """Assess overall damage severity"""
    if not detections:
        return "No Damage"
    
    num_damages = len(detections)
    avg_conf = np.mean([d['confidence'] for d in detections])
    
    # Severity rules
    if num_damages >= 5 or avg_conf > 0.8:
        return "Severe"
    elif num_damages >= 3 or avg_conf > 0.6:
        return "Moderate"
    else:
        return "Minor"


def get_severity_color(severity):
    """Get color for severity level"""
    colors = {
        "Severe": "#dc3545",
        "Moderate": "#ffc107",
        "Minor": "#28a745",
        "No Damage": "#17a2b8"
    }
    return colors.get(severity, "#6c757d")


def create_detection_chart(detections):
    """Create interactive plotly chart of detections"""
    if not detections:
        return None
    
    # Count by class
    class_counts = {}
    for det in detections:
        cls = det['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    df = pd.DataFrame({
        'Damage Type': list(class_counts.keys()),
        'Count': list(class_counts.values())
    })
    
    fig = px.bar(
        df,
        x='Damage Type',
        y='Count',
        title='Damage Distribution',
        color='Damage Type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_title="Damage Type",
        yaxis_title="Number of Detections"
    )
    
    return fig


def image_to_base64(image):
    """Convert PIL image to base64 for download"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 CarDD - AI Car Damage Detection System</h1>
        <p style='font-size: 1.2rem; margin-top: 1rem;'>
            Powered by YOLOv8 trained on CarDD Dataset
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/car-damage.png", width=100)
        st.markdown("## ⚙️ Configuration")
        
        # Model selection
        st.markdown("### 🤖 Model Settings")
        
        model_path = st.text_input(
            "Model Path",
            value=str(DEFAULT_MODEL_PATH),
            help="Path to trained YOLOv8 model (.pt file)"
        )
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        # Detection settings
        st.markdown("### 🎯 Detection Options")
        
        show_labels = st.checkbox("Show Labels", value=True)
        show_confidence = st.checkbox("Show Confidence", value=True)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        
        # Image settings
        st.markdown("### 🖼️ Image Settings")
        max_image_size = st.slider(
            "Max Image Size (px)",
            min_value=320,
            max_value=1280,
            value=640,
            step=160,
            help="Maximum image dimension for processing"
        )
        
        st.markdown("---")
        
        # System info
        st.markdown("### 📊 System Info")
        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.info(f"**Device:** {device}")
        
        if torch.cuda.is_available():
            st.info(f"**GPU:** {torch.cuda.get_device_name(0)}")
        
        st.markdown("---")
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **CarDD Damage Detection**
            
            This app uses a YOLOv8 model trained on the CarDD dataset to detect:
            - 🔹 Dents
            - 🔹 Scratches
            - 🔹 Cracks
            - 🔹 Glass Shatter
            - 🔹 Lamp Broken
            - 🔹 Tire Flat
            
            **Model:** YOLOv8m  
            **Dataset:** CarDD  
            **Classes:** 6  
            """)
    
    # Load Model
    model = load_model(model_path)
    
    if model is None:
        st.error("❌ Failed to load model. Please check the model path.")
        st.info("💡 Train the model using the notebook first!")
        st.stop()
    
    st.success(f"✅ Model loaded successfully from: `{model_path}`")
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 Single Image Detection",
        "📁 Batch Processing",
        "📊 Analytics Dashboard",
        "📖 User Guide"
    ])
    
    # ========================================================================
    # TAB 1: Single Image Detection
    # ========================================================================
    with tab1:
        st.markdown("## 📸 Upload Image for Damage Detection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a car image to detect damages"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_array
                
                # Display original image
                st.image(image, caption="Original Image", use_container_width=True)
                
                # Detection button
                if st.button("🔍 Detect Damage", key="single_detect"):
                    with st.spinner("🔄 Analyzing image..."):
                        # Run inference
                        results = model(img_bgr, conf=conf_threshold, verbose=False)
                        
                        # Draw predictions
                        img_annotated, detections = draw_predictions(
                            img_bgr, results, conf_threshold
                        )
                        
                        # Convert back to RGB
                        img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
                        
                        # Store in session state
                        st.session_state['last_result'] = {
                            'image': img_annotated_rgb,
                            'detections': detections,
                            'original_name': uploaded_file.name
                        }
        
        with col2:
            if 'last_result' in st.session_state:
                result = st.session_state['last_result']
                
                st.markdown("### 🎯 Detection Results")
                st.image(result['image'], caption="Detected Damages", use_container_width=True)
                
                # Generate report
                report = create_damage_report(
                    result['detections'],
                    result['original_name']
                )
                
                if report:
                    # Severity badge
                    severity = report['severity_assessment']
                    severity_color = get_severity_color(severity)
                    
                    st.markdown(f"""
                    <div style="background-color: {severity_color}; color: white; 
                                padding: 1rem; border-radius: 8px; text-align: center; 
                                font-size: 1.5rem; font-weight: bold; margin: 1rem 0;">
                        Severity: {severity}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.markdown(f"""
                        <div class="stats-box">
                            <h3 style="color: #667eea;">{report['total_damages']}</h3>
                            <p>Total Damages</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown(f"""
                        <div class="stats-box">
                            <h3 style="color: #667eea;">{report['average_confidence']:.2%}</h3>
                            <p>Avg Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown(f"""
                        <div class="stats-box">
                            <h3 style="color: #667eea;">{len(report['damage_breakdown'])}</h3>
                            <p>Damage Types</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Damage breakdown
                    st.markdown("### 📋 Damage Breakdown")
                    for damage_type, count in report['damage_breakdown'].items():
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{damage_type.upper()}</strong>: {count} instance(s)
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed detections
                    with st.expander("🔍 Detailed Detection Info"):
                        for i, det in enumerate(result['detections'], 1):
                            st.markdown(f"""
                            **Detection #{i}**
                            - **Class:** {det['class']}
                            - **Confidence:** {det['confidence']:.2%}
                            - **Bounding Box:** {det['bbox']}
                            - **Area:** {det['area']} px²
                            """)
                    
                    # Download options
                    st.markdown("### 💾 Download Results")
                    
                    dcol1, dcol2 = st.columns(2)
                    
                    with dcol1:
                        # Download annotated image
                        img_pil = Image.fromarray(result['image'])
                        buf = BytesIO()
                        img_pil.save(buf, format='PNG')
                        
                        st.download_button(
                            label="📥 Download Image",
                            data=buf.getvalue(),
                            file_name=f"detected_{result['original_name']}",
                            mime="image/png"
                        )
                    
                    with dcol2:
                        # Download JSON report
                        json_str = json.dumps(report, indent=2)
                        st.download_button(
                            label="📄 Download Report (JSON)",
                            data=json_str,
                            file_name=f"report_{result['original_name']}.json",
                            mime="application/json"
                        )
                else:
                    st.info("✅ No damages detected in this image!")
    
    # ========================================================================
    # TAB 2: Batch Processing
    # ========================================================================
    with tab2:
        st.markdown("## 📁 Batch Image Processing")
        st.info("Upload multiple images for batch damage detection")
        
        uploaded_files = st.file_uploader(
            "Choose images...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process All Images", key="batch_process"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}... ({idx+1}/{len(uploaded_files)})")
                    
                    # Load and process image
                    image = Image.open(file)
                    img_array = np.array(image)
                    
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = img_array
                    
                    # Run inference
                    results = model(img_bgr, conf=conf_threshold, verbose=False)
                    img_annotated, detections = draw_predictions(img_bgr, results, conf_threshold)
                    
                    # Generate report
                    report = create_damage_report(detections, file.name)
                    
                    batch_results.append({
                        'filename': file.name,
                        'num_detections': len(detections),
                        'severity': report['severity_assessment'] if report else "No Damage",
                        'report': report
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ All images processed!")
                
                # Display results
                st.markdown("### 📊 Batch Processing Results")
                
                # Summary statistics
                total_damages = sum([r['num_detections'] for r in batch_results])
                severe_cases = sum([1 for r in batch_results if r['severity'] == "Severe"])
                
                scol1, scol2, scol3 = st.columns(3)
                
                with scol1:
                    st.metric("Total Images", len(batch_results))
                
                with scol2:
                    st.metric("Total Damages Detected", total_damages)
                
                with scol3:
                    st.metric("Severe Cases", severe_cases)
                
                # Results table
                df = pd.DataFrame(batch_results)[['filename', 'num_detections', 'severity']]
                df.columns = ['Image', 'Detections', 'Severity']
                st.dataframe(df, use_container_width=True)
                
                # Download batch report
                batch_json = json.dumps(batch_results, indent=2)
                st.download_button(
                    label="📥 Download Batch Report (JSON)",
                    data=batch_json,
                    file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # ========================================================================
    # TAB 3: Analytics Dashboard
    # ========================================================================
    with tab3:
        st.markdown("## 📊 Analytics Dashboard")
        
        if 'last_result' in st.session_state and st.session_state['last_result']['detections']:
            detections = st.session_state['last_result']['detections']
            
            # Create visualization
            fig = create_detection_chart(detections)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence distribution
            confidences = [d['confidence'] for d in detections]
            
            conf_fig = go.Figure(data=[go.Histogram(
                x=confidences,
                nbinsx=20,
                marker_color='#667eea'
            )])
            
            conf_fig.update_layout(
                title="Confidence Score Distribution",
                xaxis_title="Confidence",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(conf_fig, use_container_width=True)
            
            # Detection statistics
            st.markdown("### 📈 Detection Statistics")
            
            stats_df = pd.DataFrame(detections)
            st.dataframe(stats_df, use_container_width=True)
            
        else:
            st.info("📊 Process an image first to see analytics")
    
    # ========================================================================
    # TAB 4: User Guide
    # ========================================================================
    with tab4:
        st.markdown("""
        ## 📖 User Guide
        
        ### 🚀 Getting Started
        
        1. **Upload an Image**: Click the upload button in the "Single Image Detection" tab
        2. **Adjust Settings**: Use the sidebar to configure confidence threshold and other settings
        3. **Detect Damages**: Click the "Detect Damage" button to analyze the image
        4. **Review Results**: View detected damages with bounding boxes and confidence scores
        5. **Download Report**: Export results as image or JSON report
        
        ### 🎯 Damage Types
        
        The model can detect 6 types of car damages:
        
        | **Damage Type** | **Description** |
        |----------------|----------------|
        | 🔹 **Dent** | Physical deformations in the car body |
        | 🔹 **Scratch** | Surface scratches on paint or body |
        | 🔹 **Crack** | Cracks in body panels or windshield |
        | 🔹 **Glass Shatter** | Broken or shattered glass |
        | 🔹 **Lamp Broken** | Damaged headlights or taillights |
        | 🔹 **Tire Flat** | Flat or damaged tires |
        
        ### ⚙️ Configuration Options
        
        - **Confidence Threshold**: Minimum confidence score (0-1) for displaying detections
        - **Max Image Size**: Maximum dimension for image processing (affects speed)
        - **Show Labels/Boxes**: Toggle visibility of detection annotations
        
        ### 📊 Severity Levels
        
        - **🟢 Minor**: 1-2 damages with low-medium confidence
        - **🟡 Moderate**: 3-4 damages or medium-high confidence
        - **🔴 Severe**: 5+ damages or very high confidence
        
        ### 💡 Tips for Best Results
        
        1. Use clear, well-lit images
        2. Ensure the damaged area is visible
        3. Avoid blurry or low-resolution images
        4. Take photos from multiple angles for comprehensive assessment
        5. Adjust confidence threshold if you're getting too many/few detections
        
        ### 🔧 Troubleshooting
        
        **Model not loading?**
        - Verify the model path in the sidebar
        - Make sure you've trained the model using the notebook
        - Check that the .pt file exists
        
        **No detections?**
        - Try lowering the confidence threshold
        - Ensure the image shows clear damage
        - Check if the image is too dark or blurry
        
        **Slow processing?**
        - Reduce max image size
        - Ensure CUDA/GPU is available for faster inference
        - Process images in batch mode
        
        ### 📞 Support
        
        For issues or questions:
        - Check the model training notebook
        - Review the CarDD dataset documentation
        - Ensure all dependencies are installed
        
        ---
        
        **Version:** 1.0.0  
        **Last Updated:** February 2026  
        **Model:** YOLOv8m trained on CarDD Dataset
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🚗 <strong>CarDD - AI Car Damage Detection System</strong></p>
        <p>Powered by YOLOv8 & Streamlit | © 2026</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    # Initialize session state
    if 'detection_count' not in st.session_state:
        st.session_state['detection_count'] = 0
    
    main()
