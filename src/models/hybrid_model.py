"""
Hybrid Model combining YOLOv8 for detection and U-Net for segmentation
Provides best of both worlds: real-time detection + precise segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import segmentation_models_pytorch as smp
from .yolo_model import YOLOv8CarDD


class HybridCarDDModel(nn.Module):
    """
    Hybrid model combining:
    - YOLOv8 for fast object detection
    - U-Net/DeepLabV3+ for precise segmentation
    
    This provides:
    - Real-time performance from YOLO
    - Accurate segmentation from U-Net
    - Ensemble predictions for better accuracy
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        yolo_size: str = 'x',
        seg_encoder: str = 'resnet50',
        seg_architecture: str = 'unetplusplus',
        img_size: int = 640,
        use_yolo_seg: bool = True,
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: Number of damage classes
            yolo_size: YOLOv8 model size
            seg_encoder: Segmentation encoder backbone
            seg_architecture: Segmentation architecture (unet, unetplusplus, deeplabv3plus)
            img_size: Input image size
            use_yolo_seg: Whether to use YOLOv8's segmentation
            pretrained: Use pretrained weights
        """
        super(HybridCarDDModel, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_yolo_seg = use_yolo_seg
        
        # YOLOv8 for detection
        self.yolo = YOLOv8CarDD(
            model_size=yolo_size,
            num_classes=num_classes,
            pretrained=pretrained,
            use_segmentation=use_yolo_seg,
            img_size=img_size
        )
        
        # Segmentation model
        if seg_architecture == 'unet':
            self.segmentor = smp.Unet(
                encoder_name=seg_encoder,
                encoder_weights='imagenet' if pretrained else None,
                classes=num_classes,
                activation=None
            )
        elif seg_architecture == 'unetplusplus':
            self.segmentor = smp.UnetPlusPlus(
                encoder_name=seg_encoder,
                encoder_weights='imagenet' if pretrained else None,
                classes=num_classes,
                activation=None
            )
        elif seg_architecture == 'deeplabv3plus':
            self.segmentor = smp.DeepLabV3Plus(
                encoder_name=seg_encoder,
                encoder_weights='imagenet' if pretrained else None,
                classes=num_classes,
                activation=None
            )
        else:
            raise ValueError(f"Unsupported segmentation architecture: {seg_architecture}")
        
        # Class names
        self.class_names = [
            'dent', 'scratch', 'crack',
            'glass shatter', 'lamp broken', 'tire flat'
        ]
    
    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'both'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            mode: 'detection', 'segmentation', or 'both'
        
        Returns:
            Dictionary with detection and/or segmentation outputs
        """
        outputs = {}
        
        if mode in ['detection', 'both']:
            # YOLOv8 detection
            yolo_results = self.yolo(x)
            outputs['detection'] = yolo_results
        
        if mode in ['segmentation', 'both']:
            # Semantic segmentation
            seg_logits = self.segmentor(x)
            outputs['segmentation'] = seg_logits
        
        return outputs
    
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        seg_threshold: float = 0.5,
        ensemble: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference with the hybrid model.
        
        Args:
            x: Input tensor
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            seg_threshold: Segmentation probability threshold
            ensemble: Whether to ensemble YOLO and U-Net segmentations
        
        Returns:
            Dictionary with predictions
        """
        self.eval()
        
        with torch.no_grad():
            # Detection
            yolo_preds = self.yolo.predict(x, conf=conf_threshold, iou=iou_threshold)
            
            # Segmentation
            seg_logits = self.segmentor(x)
            seg_probs = torch.sigmoid(seg_logits)
            seg_masks = (seg_probs > seg_threshold).float()
            
            # Ensemble if YOLO also provides segmentation
            if self.use_yolo_seg and ensemble:
                # Combine YOLO and U-Net segmentations
                # This is a simple averaging; more sophisticated fusion can be used
                if len(yolo_preds) > 0 and 'masks' in yolo_preds[0]:
                    # Average the segmentation masks
                    seg_masks = (seg_masks + yolo_preds[0]['masks']) / 2.0
        
        return {
            'detections': yolo_preds,
            'segmentation_logits': seg_logits,
            'segmentation_probs': seg_probs,
            'segmentation_masks': seg_masks
        }
    
    def train_detection(self, freeze_segmentor: bool = True):
        """Set model to train detection only"""
        self.yolo.train()
        if freeze_segmentor:
            self.segmentor.eval()
            for param in self.segmentor.parameters():
                param.requires_grad = False
    
    def train_segmentation(self, freeze_detector: bool = True):
        """Set model to train segmentation only"""
        self.segmentor.train()
        if freeze_detector:
            self.yolo.eval()
            for param in self.yolo.parameters():
                param.requires_grad = False
    
    def train_both(self):
        """Set model to train both components"""
        self.train()
        for param in self.parameters():
            param.requires_grad = True
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        yolo_info = self.yolo.get_model_info()
        
        seg_params = sum(p.numel() for p in self.segmentor.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        info = {
            'model_type': 'Hybrid (YOLO + Segmentation)',
            'yolo_info': yolo_info,
            'segmentation_params': seg_params,
            'total_params': total_params,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
        }
        
        return info
    
    def load_weights(self, weights_path: str, component: str = 'both'):
        """
        Load model weights.
        
        Args:
            weights_path: Path to checkpoint
            component: 'yolo', 'segmentor', or 'both'
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if component in ['yolo', 'both'] and 'yolo_state_dict' in checkpoint:
            self.yolo.load_state_dict(checkpoint['yolo_state_dict'])
        
        if component in ['segmentor', 'both'] and 'segmentor_state_dict' in checkpoint:
            self.segmentor.load_state_dict(checkpoint['segmentor_state_dict'])
    
    def save_weights(self, save_path: str, **kwargs):
        """Save model weights"""
        checkpoint = {
            'yolo_state_dict': self.yolo.state_dict(),
            'segmentor_state_dict': self.segmentor.state_dict(),
            'model_info': self.get_model_info(),
            **kwargs
        }
        torch.save(checkpoint, save_path)


def create_hybrid_model(
    num_classes: int = 6,
    yolo_size: str = 'x',
    seg_encoder: str = 'resnet50',
    **kwargs
) -> HybridCarDDModel:
    """
    Factory function to create hybrid model.
    
    Args:
        num_classes: Number of classes
        yolo_size: YOLOv8 model size
        seg_encoder: Segmentation encoder
        **kwargs: Additional arguments
    
    Returns:
        HybridCarDDModel
    """
    model = HybridCarDDModel(
        num_classes=num_classes,
        yolo_size=yolo_size,
        seg_encoder=seg_encoder,
        **kwargs
    )
    
    return model
