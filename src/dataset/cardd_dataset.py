"""
CarDD Dataset implementation for PyTorch
Supports both COCO format (detection/segmentation) and SOD format (salient object detection)
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CarDDDataset(Dataset):
    """
    CarDD Dataset for Car Damage Detection in COCO format.
    
    Supports:
    - Bounding box detection
    - Instance segmentation
    - Multi-class damage detection (6 classes)
    """
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[A.Compose] = None,
        return_masks: bool = True,
        img_size: int = 640
    ):
        """
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to COCO format annotation JSON
            transform: Albumentations transform pipeline
            return_masks: Whether to return segmentation masks
            img_size: Target image size for resizing
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_masks = return_masks
        self.img_size = img_size
        
        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_names = [cat['name'] for cat in self.categories]
        self.num_classes = len(self.categories)
        
        print(f"Loaded {len(self.image_ids)} images from {annotation_file}")
        print(f"Classes: {self.category_names}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict containing:
                - image: Tensor of shape (3, H, W)
                - boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
                - labels: Tensor of shape (N,)
                - masks: Tensor of shape (N, H, W) if return_masks=True
                - image_id: int
                - area: Tensor of shape (N,)
                - iscrowd: Tensor of shape (N,)
        """
        image_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = self.root_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Parse annotations
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Bounding box
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Category
            labels.append(ann['category_id'])
            
            # Area
            areas.append(ann['area'])
            
            # Is crowd
            iscrowd.append(ann.get('iscrowd', 0))
            
            # Segmentation mask
            if self.return_masks and 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    rles = coco_mask.frPyObjects(
                        ann['segmentation'], 
                        img_info['height'], 
                        img_info['width']
                    )
                    rle = coco_mask.merge(rles)
                elif isinstance(ann['segmentation'], dict):
                    # RLE format
                    rle = ann['segmentation']
                else:
                    continue
                    
                mask = coco_mask.decode(rle)
                masks.append(mask)
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        masks = np.array(masks, dtype=np.uint8) if masks else np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
        areas = np.array(areas, dtype=np.float32) if areas else np.zeros((0,), dtype=np.float32)
        iscrowd = np.array(iscrowd, dtype=np.int64) if iscrowd else np.zeros((0,), dtype=np.int64)
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                masks=masks if len(masks) > 0 else None,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32) if transformed['bboxes'] else np.zeros((0, 4), dtype=np.float32)
            
            if self.return_masks and len(masks) > 0:
                masks = np.array(transformed['masks'], dtype=np.uint8)
        
        # Convert to tensors
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        
        if self.return_masks and len(masks) > 0:
            target['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
        
        return image, target
    
    def get_image_info(self, idx: int) -> Dict:
        """Get image metadata"""
        image_id = self.image_ids[idx]
        return self.coco.loadImgs(image_id)[0]
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'num_images': len(self.image_ids),
            'num_classes': self.num_classes,
            'classes': self.category_names,
            'annotations_per_class': {}
        }
        
        for cat in self.categories:
            ann_ids = self.coco.getAnnIds(catIds=cat['id'])
            stats['annotations_per_class'][cat['name']] = len(ann_ids)
        
        return stats


class CarDDSODDataset(Dataset):
    """
    CarDD Salient Object Detection Dataset.
    
    Supports the SOD format with:
    - RGB images
    - Binary masks
    - Edge maps
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        img_size: int = 320
    ):
        """
        Args:
            root_dir: Root directory for SOD dataset (CarDD_SOD)
            split: 'train', 'val', or 'test'
            transform: Albumentations transform pipeline
            img_size: Target image size
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # Define paths based on split
        split_map = {
            'train': 'CarDD-TR',
            'val': 'CarDD-VAL',
            'test': 'CarDD-TE'
        }
        
        split_dir = split_map[split]
        self.img_dir = self.root_dir / split_dir / f'{split_dir}-Image'
        self.mask_dir = self.root_dir / split_dir / f'{split_dir}-Mask'
        self.edge_dir = self.root_dir / split_dir / f'{split_dir}-Edge'
        
        # Load image list
        list_file = self.root_dir / split_dir / f'{split}.lst' if split != 'train' else \
                   self.root_dir / split_dir / 'train_pair.lst'
        
        with open(list_file, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.image_list)} {split} images for SOD")
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (3, H, W)
            mask: Tensor of shape (1, H, W)
            edge: Tensor of shape (1, H, W)
        """
        img_name = self.image_list[idx].split()[0] if ' ' in self.image_list[idx] else self.image_list[idx]
        
        # Load image
        img_path = self.img_dir / f'{img_name}.jpg'
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / f'{img_name}.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Load edge
        edge_path = self.edge_dir / f'{img_name}.png'
        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                masks=[mask, edge]
            )
            image = transformed['image']
            mask, edge = transformed['masks']
        
        # Normalize masks
        mask = mask.astype(np.float32) / 255.0
        edge = edge.astype(np.float32) / 255.0
        
        # Convert to tensors
        mask = torch.from_numpy(mask).unsqueeze(0)
        edge = torch.from_numpy(edge).unsqueeze(0)
        
        return image, mask, edge


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, 0)
    
    return images, targets
