"""
Training script for Car Damage Detection models
Supports YOLOv8, Mask R-CNN, and Hybrid models
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset.cardd_dataset import CarDDDataset, collate_fn
from src.dataset.transforms import get_train_transforms, get_val_transforms
from src.models.yolo_model import create_yolo_model
from src.models.maskrcnn_model import create_maskrcnn_model
from src.models.hybrid_model import create_hybrid_model
from src.utils.metrics import COCOEvaluator
from src.utils.logger import setup_logger


class CarDDTrainer:
    """
    Trainer class for Car Damage Detection models.
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['training']['device'])
        self.epochs = self.config['training']['epochs']
        self.batch_size = self.config['training']['batch_size']
        
        # Setup logging
        self.logger = setup_logger('CarDD_Training')
        self.save_dir = Path(self.config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['experiment_name'],
                config=self.config
            )
        
        # Setup tensorboard
        if self.config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(self.config['logging']['tensorboard_dir'])
        
        # Initialize model, datasets, and optimizer
        self._setup_model()
        self._setup_datasets()
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.global_step = 0
    
    def _setup_model(self):
        """Initialize model based on configuration"""
        model_arch = self.config['model']['architecture']
        num_classes = self.config['dataset']['num_classes']
        
        if model_arch == 'yolov8':
            self.model = create_yolo_model(
                model_size=self.config['model']['backbone'].replace('yolov8', ''),
                num_classes=num_classes,
                pretrained=self.config['model']['pretrained'],
                use_segmentation=self.config['model']['use_segmentation']
            )
        elif model_arch == 'mask_rcnn':
            self.model = create_maskrcnn_model(
                num_classes=num_classes + 1,  # +1 for background
                pretrained=self.config['model']['pretrained']
            )
        elif model_arch == 'hybrid':
            self.model = create_hybrid_model(
                num_classes=num_classes,
                yolo_size=self.config['model']['backbone'].replace('yolov8', ''),
                pretrained=self.config['model']['pretrained']
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_arch}")
        
        self.model = self.model.to(self.device)
        self.logger.info(f"Model initialized: {model_arch}")
        self.logger.info(f"Model info: {self.model.get_model_info()}")
    
    def _setup_datasets(self):
        """Setup training and validation datasets"""
        # Training dataset
        train_transform = get_train_transforms(
            img_size=self.config['model'].get('yolo', {}).get('img_size', 640)
        )
        
        self.train_dataset = CarDDDataset(
            root_dir=self.config['dataset']['train_img_dir'],
            annotation_file=self.config['dataset']['train_ann'],
            transform=train_transform,
            return_masks=self.config['model']['use_segmentation']
        )
        
        # Validation dataset
        val_transform = get_val_transforms(
            img_size=self.config['model'].get('yolo', {}).get('img_size', 640)
        )
        
        self.val_dataset = CarDDDataset(
            root_dir=self.config['dataset']['val_img_dir'],
            annotation_file=self.config['dataset']['val_ann'],
            transform=val_transform,
            return_masks=self.config['model']['use_segmentation']
        )
        
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.logger.info(f"Train dataset: {len(self.train_dataset)} images")
        self.logger.info(f"Val dataset: {len(self.val_dataset)} images")
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        opt_config = self.config['training']['optimizer']
        opt_type = opt_config['type']
        
        if opt_type == 'Adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_type == 'AdamW':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_type == 'SGD':
            self.optimizer = SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config['momentum'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")
        
        self.logger.info(f"Optimizer initialized: {opt_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        sched_config = self.config['training']['scheduler']
        sched_type = sched_config['type']
        
        if sched_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=sched_config['min_lr']
            )
        elif sched_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif sched_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=10
            )
        elif sched_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['optimizer']['lr'],
                epochs=self.epochs,
                steps_per_epoch=len(self.train_loader)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_type}")
        
        self.logger.info(f"Scheduler initialized: {sched_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.epochs}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            if isinstance(self.model, type(create_maskrcnn_model())):
                # Mask R-CNN returns loss dict directly
                loss_dict = self.model.train_step(list(images), targets)
                losses = sum(loss for loss in loss_dict.values())
            else:
                # For YOLO, use its training method
                # This is handled by YOLO's own training loop
                pass
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update progress bar
            epoch_loss += losses.item()
            progress_bar.set_postfix({'loss': losses.item()})
            
            # Log to wandb/tensorboard
            if self.global_step % self.config['logging']['log_interval'] == 0:
                if self.config['logging']['use_wandb']:
                    wandb.log({'train/loss': losses.item()}, step=self.global_step)
                if self.config['logging']['use_tensorboard']:
                    self.writer.add_scalar('train/loss', losses.item(), self.global_step)
            
            self.global_step += 1
        
        return {'loss': epoch_loss / len(self.train_loader)}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        evaluator = COCOEvaluator(self.val_dataset.coco)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                
                # Get predictions
                predictions = self.model.predict(images)
                
                # Add to evaluator
                for pred, target in zip(predictions, targets):
                    evaluator.update(pred, target)
        
        # Compute metrics
        metrics = evaluator.compute()
        
        return metrics
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if (epoch + 1) % 5 == 0:
                val_metrics = self.validate()
                self.logger.info(f"Epoch {epoch + 1} - Val mAP: {val_metrics.get('mAP', 0):.4f}")
                
                # Log to wandb/tensorboard
                if self.config['logging']['use_wandb']:
                    wandb.log({f'val/{k}': v for k, v in val_metrics.items()}, step=epoch)
                if self.config['logging']['use_tensorboard']:
                    for k, v in val_metrics.items():
                        self.writer.add_scalar(f'val/{k}', v, epoch)
                
                # Save best model
                if val_metrics.get('mAP', 0) > self.best_map:
                    self.best_map = val_metrics.get('mAP', 0)
                    self.save_checkpoint('best_model.pth')
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config['training']['save_period'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('mAP', 0))
            else:
                self.scheduler.step()
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.save_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CarDD Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['yolov8', 'maskrcnn', 'hybrid'],
                        help='Model architecture (overrides config)')
    parser.add_argument('--backbone', type=str,
                        help='Model backbone (e.g., yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.model:
        config['model']['architecture'] = args.model
    if args.backbone:
        config['model']['backbone'] = args.backbone
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.device:
        config['training']['device'] = args.device
    
    # Save modified config temporarily
    temp_config_path = Path('config/temp_config.yaml')
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Initialize trainer
    trainer = CarDDTrainer(str(temp_config_path))
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_map = checkpoint['best_map']
        trainer.logger.info(f"Resumed training from epoch {trainer.current_epoch}")
    
    trainer.train()


if __name__ == '__main__':
    main()
