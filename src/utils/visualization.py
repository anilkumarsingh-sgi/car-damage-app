"""
Visualization utilities for Car Damage Detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import torch
from pathlib import Path


# Define colors for each damage class
CLASS_COLORS = {
    0: (255, 0, 0),      # dent - red
    1: (0, 255, 0),      # scratch - green
    2: (0, 0, 255),      # crack - blue
    3: (255, 255, 0),    # glass shatter - yellow
    4: (255, 0, 255),    # lamp broken - magenta
    5: (0, 255, 255),    # tire flat - cyan
}

CLASS_NAMES = [
    'dent', 'scratch', 'crack',
    'glass shatter', 'lamp broken', 'tire flat'
]


def visualize_predictions(
    image: np.ndarray,
    predictions: Dict[str, torch.Tensor],
    conf_threshold: float = 0.5,
    show_labels: bool = True,
    show_conf: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Visualize detection predictions on image.
    
    Args:
        image: Input image (H, W, 3) in RGB
        predictions: Prediction dictionary with boxes, labels, scores
        conf_threshold: Confidence threshold
        show_labels: Whether to show class labels
        show_conf: Whether to show confidence scores
        line_thickness: Box line thickness
        font_scale: Font scale for text
    
    Returns:
        Annotated image
    """
    img = image.copy()
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    # Denormalize if needed
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
        labels = labels.cpu().numpy()
        scores = scores.cpu().numpy()
    
    # Filter by confidence
    keep = scores > conf_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Get color for this class
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        
        # Prepare label text
        if show_labels and show_conf:
            text = f"{CLASS_NAMES[label]}: {score:.2f}"
        elif show_labels:
            text = CLASS_NAMES[label]
        elif show_conf:
            text = f"{score:.2f}"
        else:
            text = ""
        
        if text:
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                img,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    # Draw masks if available
    if 'masks' in predictions and len(predictions['masks']) > 0:
        masks = predictions['masks']
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        masks = masks[keep]
        
        for mask, label in zip(masks, labels):
            if mask.ndim == 3:
                mask = mask[0]
            
            # Resize mask to image size
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
            
            # Create colored mask
            color = np.array(CLASS_COLORS.get(label, (255, 255, 255)))
            colored_mask = np.zeros_like(img)
            colored_mask[mask > 0] = color
            
            # Blend with image
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.4, 0)
    
    return img


def visualize_batch(
    images: torch.Tensor,
    predictions: List[Dict[str, torch.Tensor]],
    save_path: Optional[str] = None,
    max_images: int = 16
) -> None:
    """
    Visualize a batch of predictions.
    
    Args:
        images: Batch of images (B, 3, H, W)
        predictions: List of prediction dictionaries
        save_path: Path to save visualization
        max_images: Maximum number of images to show
    """
    n_images = min(len(images), max_images)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_images):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Visualize predictions
        img = visualize_predictions(img, predictions[i])
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i + 1}')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_metrics(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training metrics over time.
    
    Args:
        metrics_history: Dictionary mapping metric names to lists of values
        save_path: Path to save plot
    """
    n_metrics = len(metrics_history)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, (metric_name, values) in enumerate(metrics_history.items()):
        axes[i].plot(values, marker='o', linewidth=2)
        axes[i].set_title(metric_name, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel('Value', fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_comparison_plot(
    image: np.ndarray,
    ground_truth: Dict,
    prediction: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Create side-by-side comparison of ground truth and prediction.
    
    Args:
        image: Original image
        ground_truth: Ground truth annotations
        prediction: Model predictions
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Ground truth
    gt_img = visualize_predictions(
        image.copy(),
        ground_truth,
        conf_threshold=0.0,
        show_conf=False
    )
    axes[0].imshow(gt_img)
    axes[0].set_title('Ground Truth', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Prediction
    pred_img = visualize_predictions(
        image.copy(),
        prediction,
        conf_threshold=0.5,
        show_conf=True
    )
    axes[1].imshow(pred_img)
    axes[1].set_title('Prediction', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
