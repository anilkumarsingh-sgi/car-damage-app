"""
YOLOv8 implementation for Car Damage Detection
Uses Ultralytics YOLOv8 with custom modifications for CarDD
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


class YOLOv8CarDD(nn.Module):
    """
    YOLOv8 model customized for CarDD dataset.
    
    Features:
    - State-of-the-art real-time detection
    - Support for instance segmentation
    - Efficient backbone architecture
    - Post-processing with NMS
    """
    
    def __init__(
        self,
        model_size: str = 'x',  # n, s, m, l, x
        num_classes: int = 6,
        pretrained: bool = True,
        use_segmentation: bool = True,
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Args:
            model_size: YOLOv8 model size (n, s, m, l, x)
            num_classes: Number of damage classes
            pretrained: Whether to use pretrained weights
            use_segmentation: Whether to use instance segmentation
            img_size: Input image size
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
        """
        super(YOLOv8CarDD, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_segmentation = use_segmentation
        
        # Initialize YOLOv8
        model_type = f'yolov8{model_size}'
        if use_segmentation:
            model_type += '-seg'
        
        if pretrained:
            model_type += '.pt'
        
        self.model = YOLO(model_type)
        
        # Class names for CarDD
        self.class_names = [
            'dent', 'scratch', 'crack', 
            'glass shatter', 'lamp broken', 'tire flat'
        ]
    
    def forward(self, x: torch.Tensor) -> List[Dict]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            List of dictionaries with predictions
        """
        # YOLOv8 forward pass
        results = self.model(x, verbose=False)
        
        # Parse results
        outputs = []
        for result in results:
            output = {
                'boxes': result.boxes.xyxy.cpu(),
                'scores': result.boxes.conf.cpu(),
                'labels': result.boxes.cls.cpu().long(),
            }
            
            if self.use_segmentation and hasattr(result, 'masks') and result.masks is not None:
                output['masks'] = result.masks.data.cpu()
            
            outputs.append(output)
        
        return outputs
    
    def train_model(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        device: Union[str, int] = 'cuda',
        save_dir: str = 'runs/train',
        **kwargs
    ):
        """
        Train the YOLOv8 model.
        
        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of training epochs
            batch_size: Batch size
            device: Device to train on
            save_dir: Directory to save results
            **kwargs: Additional training arguments
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=self.img_size,
            device=device,
            project=save_dir,
            name='yolov8_cardd',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=2.0,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            **kwargs
        )
        
        return results
    
    def predict(
        self,
        source: Union[str, np.ndarray, torch.Tensor],
        save: bool = False,
        save_dir: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        **kwargs
    ):
        """
        Run inference on images.
        
        Args:
            source: Image source (path, numpy array, or tensor)
            save: Whether to save results
            save_dir: Directory to save results
            conf: Confidence threshold (uses default if None)
            iou: IoU threshold (uses default if None)
            **kwargs: Additional inference arguments
        
        Returns:
            Inference results
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        
        results = self.model.predict(
            source=source,
            save=save,
            project=save_dir,
            name='predictions',
            exist_ok=True,
            conf=conf,
            iou=iou,
            imgsz=self.img_size,
            **kwargs
        )
        
        return results
    
    def evaluate(
        self,
        data_yaml: str,
        split: str = 'val',
        **kwargs
    ):
        """
        Evaluate the model on validation/test set.
        
        Args:
            data_yaml: Path to data configuration YAML
            split: Dataset split to evaluate on
            **kwargs: Additional evaluation arguments
        
        Returns:
            Evaluation metrics
        """
        results = self.model.val(
            data=data_yaml,
            split=split,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            **kwargs
        )
        
        return results
    
    def export(
        self,
        format: str = 'onnx',
        dynamic: bool = True,
        simplify: bool = True,
        **kwargs
    ):
        """
        Export model to different formats.
        
        Args:
            format: Export format (onnx, torchscript, coreml, etc.)
            dynamic: Dynamic axes for ONNX
            simplify: Simplify ONNX model
            **kwargs: Additional export arguments
        
        Returns:
            Path to exported model
        """
        export_path = self.model.export(
            format=format,
            dynamic=dynamic,
            simplify=simplify,
            imgsz=self.img_size,
            **kwargs
        )
        
        return export_path
    
    def load_weights(self, weights_path: str):
        """Load model weights from checkpoint"""
        self.model = YOLO(weights_path)
    
    def save_weights(self, save_path: str):
        """Save model weights"""
        self.model.save(save_path)
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics"""
        info = {
            'model_type': type(self.model).__name__,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'use_segmentation': self.use_segmentation,
            'class_names': self.class_names,
        }
        
        return info


def create_yolo_model(
    model_size: str = 'x',
    num_classes: int = 6,
    pretrained: bool = True,
    **kwargs
) -> YOLOv8CarDD:
    """
    Factory function to create YOLOv8 model.
    
    Args:
        model_size: Model size (n, s, m, l, x)
        num_classes: Number of classes
        pretrained: Use pretrained weights
        **kwargs: Additional model arguments
    
    Returns:
        YOLOv8CarDD model
    """
    model = YOLOv8CarDD(
        model_size=model_size,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
    
    return model
