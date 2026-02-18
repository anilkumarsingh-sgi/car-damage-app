"""
Mask R-CNN implementation for Car Damage Detection
Uses torchvision's Mask R-CNN with custom modifications
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MaskRCNNCarDD(nn.Module):
    """
    Mask R-CNN model for CarDD dataset.
    
    Features:
    - Instance segmentation and bounding box detection
    - ResNet backbone with FPN
    - ROI pooling for precise localization
    - Support for transfer learning
    """
    
    def __init__(
        self,
        num_classes: int = 7,  # 6 damage classes + background
        backbone: str = 'resnet50',
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        min_size: int = 800,
        max_size: int = 1333,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes (including background)
            backbone: Backbone architecture
            pretrained: Use COCO pretrained weights
            pretrained_backbone: Use ImageNet pretrained backbone
            trainable_backbone_layers: Number of trainable backbone layers
            min_size: Minimum image size
            max_size: Maximum image size
        """
        super(MaskRCNNCarDD, self).__init__()
        
        self.num_classes = num_classes
        self.min_size = min_size
        self.max_size = max_size
        
        # Load pretrained Mask R-CNN
        if backbone == 'resnet50':
            self.model = maskrcnn_resnet50_fpn(
                pretrained=pretrained,
                pretrained_backbone=pretrained_backbone,
                trainable_backbone_layers=trainable_backbone_layers,
                min_size=min_size,
                max_size=max_size,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace the box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
        
        # Replace the mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        
        # Class names
        self.class_names = [
            'background', 'dent', 'scratch', 'crack',
            'glass shatter', 'lamp broken', 'tire flat'
        ]
    
    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the model.
        
        Args:
            images: List of images, each of shape (3, H, W)
            targets: List of target dictionaries (during training)
        
        Returns:
            If training: losses dict
            If inference: list of prediction dictionaries
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        return self.model(images, targets)
    
    def predict(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference on images.
        
        Args:
            images: Input images
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
        
        Returns:
            List of prediction dictionaries
        """
        self.eval()
        
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = [images]
            elif images.dim() == 4:
                images = [img for img in images]
        
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter by confidence
        filtered_predictions = []
        for pred in predictions:
            keep = pred['scores'] > conf_threshold
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep],
                'masks': pred['masks'][keep]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def train_step(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            images: List of input images
            targets: List of target dictionaries
        
        Returns:
            Dictionary of losses
        """
        self.train()
        loss_dict = self.model(images, targets)
        return loss_dict
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'Mask R-CNN',
            'backbone': 'ResNet50-FPN',
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'class_names': self.class_names,
        }
        
        return info
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
    
    def load_weights(self, weights_path: str, strict: bool = True):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            self.load_state_dict(checkpoint, strict=strict)
    
    def save_weights(self, save_path: str, **kwargs):
        """Save model weights"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            **kwargs
        }
        torch.save(checkpoint, save_path)


def create_maskrcnn_model(
    num_classes: int = 7,
    pretrained: bool = True,
    **kwargs
) -> MaskRCNNCarDD:
    """
    Factory function to create Mask R-CNN model.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Use pretrained weights
        **kwargs: Additional model arguments
    
    Returns:
        MaskRCNNCarDD model
    """
    model = MaskRCNNCarDD(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
    
    return model
