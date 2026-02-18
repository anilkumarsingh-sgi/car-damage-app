"""
Evaluation metrics for Car Damage Detection
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import json


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: Box in [x1, y1, x2, y2] format
        box2: Box in [x1, y1, x2, y2] format
    
    Returns:
        IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def compute_ap(
    precisions: np.ndarray,
    recalls: np.ndarray
) -> float:
    """
    Compute Average Precision (AP).
    
    Args:
        precisions: Array of precision values
        recalls: Array of recall values
    
    Returns:
        AP value
    """
    # Append sentinel values
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    
    # Compute the precision envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Compute AP
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def compute_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 6
) -> Dict[str, float]:
    """
    Compute detection metrics (mAP, precision, recall, F1).
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    all_precisions = []
    all_recalls = []
    all_aps = []
    
    for cls in range(1, num_classes + 1):
        # Get all predictions and targets for this class
        cls_preds = []
        cls_targets = []
        
        for pred in predictions:
            mask = pred['labels'] == cls
            if mask.sum() > 0:
                cls_preds.append({
                    'boxes': pred['boxes'][mask],
                    'scores': pred['scores'][mask]
                })
        
        for target in targets:
            mask = target['labels'] == cls
            if mask.sum() > 0:
                cls_targets.append({
                    'boxes': target['boxes'][mask]
                })
        
        if len(cls_preds) == 0 or len(cls_targets) == 0:
            continue
        
        # Compute TP, FP, FN
        tp = 0
        fp = 0
        fn = len(cls_targets)
        
        for pred in cls_preds:
            pred_boxes = pred['boxes'].numpy() if isinstance(pred['boxes'], torch.Tensor) else pred['boxes']
            pred_scores = pred['scores'].numpy() if isinstance(pred['scores'], torch.Tensor) else pred['scores']
            
            # Sort by confidence
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            
            matched = np.zeros(len(pred_boxes), dtype=bool)
            
            for target in cls_targets:
                target_boxes = target['boxes'].numpy() if isinstance(target['boxes'], torch.Tensor) else target['boxes']
                
                for i, pred_box in enumerate(pred_boxes):
                    if matched[i]:
                        continue
                    
                    for target_box in target_boxes:
                        iou = compute_iou(pred_box, target_box)
                        if iou >= iou_threshold:
                            tp += 1
                            matched[i] = True
                            fn -= 1
                            break
            
            fp += (~matched).sum()
        
        # Compute precision and recall
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
    
    # Compute mean metrics
    mean_precision = np.mean(all_precisions) if all_precisions else 0.0
    mean_recall = np.mean(all_recalls) if all_recalls else 0.0
    f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall + 1e-6)
    
    metrics = {
        'precision': mean_precision,
        'recall': mean_recall,
        'f1_score': f1_score,
        'mAP': np.mean(all_aps) if all_aps else 0.0
    }
    
    return metrics


class COCOEvaluator:
    """
    COCO-style evaluation for object detection and instance segmentation.
    """
    
    def __init__(self, coco_gt: COCO, iou_type: str = 'bbox'):
        """
        Args:
            coco_gt: COCO ground truth object
            iou_type: 'bbox' or 'segm'
        """
        self.coco_gt = coco_gt
        self.iou_type = iou_type
        self.results = []
    
    def update(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ):
        """
        Add a prediction to the evaluator.
        
        Args:
            prediction: Prediction dictionary
            target: Target dictionary
        """
        image_id = target['image_id'].item()
        
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            
            result = {
                'image_id': image_id,
                'category_id': int(label),
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': float(score)
            }
            
            if 'masks' in prediction:
                # Add segmentation
                mask = prediction['masks'][0].cpu().numpy()
                # Convert mask to RLE (Run Length Encoding)
                # This is simplified; use pycocotools.mask for production
                result['segmentation'] = mask.tolist()
            
            self.results.append(result)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute COCO metrics.
        
        Returns:
            Dictionary of metrics (AP, AP50, AP75, etc.)
        """
        if not self.results:
            return {
                'mAP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'APs': 0.0,
                'APm': 0.0,
                'APl': 0.0
            }
        
        # Create results file
        coco_dt = self.coco_gt.loadRes(self.results)
        
        # Run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, self.iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'mAP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
            'APs': coco_eval.stats[3],
            'APm': coco_eval.stats[4],
            'APl': coco_eval.stats[5],
        }
        
        return metrics
    
    def reset(self):
        """Reset the evaluator"""
        self.results = []


class SegmentationMetrics:
    """
    Metrics for semantic segmentation.
    """
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update confusion matrix.
        
        Args:
            pred: Predicted segmentation (B, H, W) or (B, C, H, W)
            target: Target segmentation (B, H, W)
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
        
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        mask = (target >= 0) & (target < self.num_classes)
        
        indices = self.num_classes * target[mask] + pred[mask]
        self.confusion_matrix += np.bincount(
            indices.astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute segmentation metrics.
        
        Returns:
            Dictionary with IoU, Dice, Pixel Accuracy
        """
        # Pixel Accuracy
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        
        # Mean IoU
        iou_per_class = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            np.diag(self.confusion_matrix) + 1e-6
        )
        mean_iou = np.nanmean(iou_per_class)
        
        # Dice coefficient
        dice_per_class = 2 * np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) + 1e-6
        )
        mean_dice = np.nanmean(dice_per_class)
        
        metrics = {
            'pixel_accuracy': acc,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'iou_per_class': iou_per_class.tolist(),
            'dice_per_class': dice_per_class.tolist()
        }
        
        return metrics
    
    def reset(self):
        """Reset confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
