"""
Simple ResNet50-based Car Damage Classifier
Uses pretrained ResNet50 for damage detection/classification
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple


CLASS_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']


class ResNet50DamageClassifier(nn.Module):
    """
    ResNet50-based damage classifier
    Can classify entire image or image patches for damage type
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        pretrained_path: str = None,
        device: str = 'cuda'
    ):
        """
        Args:
            num_classes: Number of damage classes
            pretrained_path: Path to resnet50-19c8e357.pth or None to download
            device: Device to run on
        """
        super(ResNet50DamageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load ResNet50
        self.model = models.resnet50(pretrained=False)
        
        # Load weights if provided
        if pretrained_path:
            try:
                state_dict = torch.load(pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Loaded ResNet50 from {pretrained_path}")
            except Exception as e:
                print(f"⚠️ Could not load {pretrained_path}, using PyTorch pretrained weights")
                self.model = models.resnet50(pretrained=True)
        else:
            # Use PyTorch's pretrained weights
            self.model = models.resnet50(pretrained=True)
        
        # Replace final layer for damage classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for ResNet50"""
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict_image(
        self,
        image: Image.Image,
        top_k: int = 3
    ) -> Dict[str, float]:
        """
        Classify entire image for damage type
        
        Args:
            image: PIL Image
            top_k: Return top K predictions
        
        Returns:
            Dictionary with class names and probabilities
        """
        # Preprocess
        img_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        
        # Get top K predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes))
        
        results = {}
        for prob, idx in zip(top_probs, top_indices):
            class_name = CLASS_NAMES[idx.item()]
            results[class_name] = float(prob.item())
        
        return results
    
    def predict_with_sliding_window(
        self,
        image: Image.Image,
        window_size: int = 224,
        stride: int = 112,
        conf_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Detect damage using sliding window approach
        
        Args:
            image: PIL Image
            window_size: Size of sliding window
            stride: Stride for sliding window
            conf_threshold: Confidence threshold
        
        Returns:
            List of detections with bounding boxes and classes
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        detections = []
        
        # Slide window over image
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                # Extract patch
                patch = image.crop((x, y, x + window_size, y + window_size))
                
                # Classify patch
                results = self.predict_image(patch, top_k=1)
                
                # If confident detection, add to results
                for class_name, prob in results.items():
                    if prob > conf_threshold:
                        detections.append({
                            'bbox': [x, y, x + window_size, y + window_size],
                            'class': class_name,
                            'confidence': prob,
                            'center': [(x + x + window_size) / 2, (y + y + window_size) / 2]
                        })
        
        # Non-maximum suppression (simple version)
        detections = self._simple_nms(detections, iou_threshold=0.5)
        
        return detections
    
    def _simple_nms(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Simple non-maximum suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._calculate_iou(best['bbox'], det['bbox']) < iou_threshold
                or best['class'] != det['class']
            ]
        
        return keep
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


def create_resnet50_classifier(
    num_classes: int = 6,
    pretrained_path: str = None,
    device: str = 'cuda'
) -> ResNet50DamageClassifier:
    """
    Create ResNet50 damage classifier
    
    Args:
        num_classes: Number of damage classes
        pretrained_path: Path to resnet50-19c8e357.pth
        device: Device to use
    
    Returns:
        ResNet50DamageClassifier instance
    """
    return ResNet50DamageClassifier(
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        device=device
    )


if __name__ == '__main__':
    # Test the classifier
    classifier = create_resnet50_classifier(
        pretrained_path='resnet50-19c8e357.pth'
    )
    
    # Test with dummy image
    from PIL import Image
    dummy_img = Image.new('RGB', (640, 480), color='red')
    
    # Classification
    results = classifier.predict_image(dummy_img)
    print("Classification results:", results)
    
    # Detection with sliding window
    detections = classifier.predict_with_sliding_window(dummy_img)
    print(f"Found {len(detections)} detections")
