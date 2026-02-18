"""
Evaluation script for Car Damage Detection models
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent))

from src.dataset.cardd_dataset import CarDDDataset, collate_fn
from src.dataset.transforms import get_val_transforms
from src.models.yolo_model import create_yolo_model
from src.models.maskrcnn_model import create_maskrcnn_model
from src.models.hybrid_model import create_hybrid_model
from src.utils.metrics import COCOEvaluator, SegmentationMetrics
from src.utils.logger import setup_logger
from src.utils.visualization import plot_metrics, plot_confusion_matrix
from prettytable import PrettyTable


def evaluate_model(
    model_path: str,
    config_path: str,
    split: str = 'test',
    device: str = 'cuda',
    save_results: bool = True
):
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        split: Dataset split to evaluate on
        device: Device to run evaluation on
        save_results: Whether to save evaluation results
    """
    # Setup logger
    logger = setup_logger('CarDD_Evaluation')
    logger.info(f"Evaluating model: {model_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_arch = config['model']['architecture']
    num_classes = config['dataset']['num_classes']
    
    if model_arch == 'yolov8':
        model = create_yolo_model(
            model_size=config['model']['backbone'].replace('yolov8', ''),
            num_classes=num_classes,
            pretrained=False
        )
    elif model_arch == 'mask_rcnn':
        model = create_maskrcnn_model(
            num_classes=num_classes + 1,
            pretrained=False
        )
    elif model_arch == 'hybrid':
        model = create_hybrid_model(
            num_classes=num_classes,
            pretrained=False
        )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    
    # Load dataset
    ann_file = config['dataset'][f'{split}_ann']
    img_dir = config['dataset'][f'{split}_img_dir']
    
    dataset = CarDDDataset(
        root_dir=img_dir,
        annotation_file=ann_file,
        transform=get_val_transforms(640),
        return_masks=config['model']['use_segmentation']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} images")
    
    # Initialize evaluators
    coco_evaluator = COCOEvaluator(dataset.coco, iou_type='bbox')
    seg_evaluator = SegmentationMetrics(num_classes) if config['model']['use_segmentation'] else None
    
    # Evaluation loop
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            # Get predictions
            predictions = model.predict(images)
            
            # Update evaluators
            for pred, target in zip(predictions, targets):
                coco_evaluator.update(pred, target)
                
                if seg_evaluator is not None and 'masks' in pred:
                    seg_evaluator.update(pred['masks'], target['masks'])
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    detection_metrics = coco_evaluator.compute()
    
    # Print results
    print("\n" + "="*50)
    print("DETECTION METRICS")
    print("="*50)
    
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.align["Metric"] = "l"
    table.align["Value"] = "r"
    
    for metric, value in detection_metrics.items():
        table.add_row([metric, f"{value:.4f}"])
    
    print(table)
    
    # Segmentation metrics
    if seg_evaluator is not None:
        seg_metrics = seg_evaluator.compute()
        
        print("\n" + "="*50)
        print("SEGMENTATION METRICS")
        print("="*50)
        
        seg_table = PrettyTable()
        seg_table.field_names = ["Metric", "Value"]
        seg_table.align["Metric"] = "l"
        seg_table.align["Value"] = "r"
        
        for metric, value in seg_metrics.items():
            if not isinstance(value, list):
                seg_table.add_row([metric, f"{value:.4f}"])
        
        print(seg_table)
    
    # Per-class metrics
    print("\n" + "="*50)
    print("PER-CLASS METRICS")
    print("="*50)
    
    # Save results
    if save_results:
        results_dir = Path('evaluation_results')
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'model_path': model_path,
            'split': split,
            'detection_metrics': detection_metrics,
        }
        
        if seg_evaluator is not None:
            results['segmentation_metrics'] = seg_metrics
        
        # Save as JSON
        with open(results_dir / f'evaluation_{split}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to: {results_dir}")
    
    return detection_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate CarDD Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run evaluation on')
    parser.add_argument('--save', action='store_true',
                        help='Save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        config_path=args.config,
        split=args.split,
        device=args.device,
        save_results=args.save
    )


if __name__ == '__main__':
    main()
