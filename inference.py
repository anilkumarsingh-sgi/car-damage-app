"""
Inference script for Car Damage Detection
Supports single image, batch images, video, and webcam inference
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Union, List, Optional, Dict
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.yolo_model import create_yolo_model
from src.models.maskrcnn_model import create_maskrcnn_model
from src.models.hybrid_model import create_hybrid_model
from src.utils.visualization import visualize_predictions, CLASS_NAMES
from src.utils.logger import setup_logger


class CarDDInference:
    """
    Inference class for Car Damage Detection.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = 'config/config.yaml',
        device: str = 'cuda'
    ):
        """
        Args:
            model_path: Path to model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger('CarDD_Inference')
        
        # Load model
        self._load_model(model_path)
        
        self.logger.info(f"Model loaded from: {model_path}")
        self.logger.info(f"Running on device: {self.device}")
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        model_arch = self.config['model']['architecture']
        num_classes = self.config['dataset']['num_classes']
        
        # Create model
        if model_arch == 'yolov8':
            self.model = create_yolo_model(
                model_size=self.config['model']['backbone'].replace('yolov8', ''),
                num_classes=num_classes,
                pretrained=False
            )
        elif model_arch == 'mask_rcnn':
            self.model = create_maskrcnn_model(
                num_classes=num_classes + 1,
                pretrained=False
            )
        elif model_arch == 'hybrid':
            self.model = create_hybrid_model(
                num_classes=num_classes,
                pretrained=False
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_arch}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict_image(
        self,
        image_path: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        save_path: Optional[str] = None,
        visualize: bool = True
    ) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to image or numpy array
            conf_threshold: Confidence threshold
            save_path: Path to save annotated image
            visualize: Whether to visualize results
        
        Returns:
            Prediction dictionary
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Prepare image for model
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model.predict(
                img_tensor,
                conf_threshold=conf_threshold
            )
        
        # Process predictions
        if isinstance(predictions, list):
            prediction = predictions[0]
        else:
            prediction = predictions
        
        # Visualize if requested
        if visualize:
            annotated_image = visualize_predictions(
                image,
                prediction,
                conf_threshold=conf_threshold,
                show_labels=True,
                show_conf=True
            )
            
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                self.logger.info(f"Saved annotated image to: {save_path}")
        
        return prediction
    
    def predict_batch(
        self,
        image_dir: str,
        output_dir: str,
        conf_threshold: float = 0.25,
        extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ):
        """
        Run inference on a directory of images.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            conf_threshold: Confidence threshold
            extensions: Allowed image extensions
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
        
        self.logger.info(f"Found {len(image_files)} images")
        
        # Process each image
        for img_path in tqdm(image_files, desc='Processing images'):
            save_path = output_dir / f'{img_path.stem}_annotated{img_path.suffix}'
            self.predict_image(img_path, conf_threshold, save_path=str(save_path))
    
    def predict_video(
        self,
        video_path: str,
        output_path: str,
        conf_threshold: float = 0.25,
        show_preview: bool = False
    ):
        """
        Run inference on a video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            conf_threshold: Confidence threshold
            show_preview: Whether to show live preview
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.logger.info(f"Processing video: {total_frames} frames @ {fps} FPS")
        
        # Process frames
        with tqdm(total=total_frames, desc='Processing video') as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference
                prediction = self.predict_image(
                    frame_rgb,
                    conf_threshold=conf_threshold,
                    visualize=False
                )
                
                # Visualize
                annotated_frame = visualize_predictions(
                    frame_rgb,
                    prediction,
                    conf_threshold=conf_threshold
                )
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(annotated_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('CarDD Inference', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        self.logger.info(f"Saved output video to: {output_path}")
    
    def predict_webcam(
        self,
        camera_id: int = 0,
        conf_threshold: float = 0.25
    ):
        """
        Run real-time inference on webcam feed.
        
        Args:
            camera_id: Camera device ID
            conf_threshold: Confidence threshold
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open camera {camera_id}")
            return
        
        self.logger.info("Starting webcam inference. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            prediction = self.predict_image(
                frame_rgb,
                conf_threshold=conf_threshold,
                visualize=False
            )
            
            # Visualize
            annotated_frame = visualize_predictions(
                frame_rgb,
                prediction,
                conf_threshold=conf_threshold
            )
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Display
            cv2.imshow('CarDD Webcam Inference', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='CarDD Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, directory, video, or "webcam"')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Path to save results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--show', action='store_true',
                        help='Show preview for video/webcam')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = CarDDInference(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # Run inference based on source type
    if args.source == 'webcam':
        inferencer.predict_webcam(conf_threshold=args.conf)
    elif os.path.isfile(args.source):
        # Check if video or image
        ext = Path(args.source).suffix.lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            output_path = str(Path(args.output) / 'output_video.mp4')
            inferencer.predict_video(
                args.source,
                output_path,
                conf_threshold=args.conf,
                show_preview=args.show
            )
        else:
            output_path = str(Path(args.output) / f'{Path(args.source).stem}_annotated{ext}')
            inferencer.predict_image(
                args.source,
                conf_threshold=args.conf,
                save_path=output_path
            )
    elif os.path.isdir(args.source):
        inferencer.predict_batch(
            args.source,
            args.output,
            conf_threshold=args.conf
        )
    else:
        print(f"Invalid source: {args.source}")


if __name__ == '__main__':
    main()
