"""
Data preparation and analysis script for CarDD dataset
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from prettytable import PrettyTable
import cv2


def analyze_dataset(annotation_file: str, split_name: str = 'train'):
    """
    Analyze CarDD dataset statistics.
    
    Args:
        annotation_file: Path to COCO format annotation file
        split_name: Name of the split (train/val/test)
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {split_name.upper()} Dataset")
    print(f"{'='*60}\n")
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Basic statistics
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    num_categories = len(data['categories'])
    
    print(f"Total Images: {num_images}")
    print(f"Total Annotations: {num_annotations}")
    print(f"Number of Classes: {num_categories}")
    print(f"Average Annotations per Image: {num_annotations / num_images:.2f}\n")
    
    # Category statistics
    category_counts = Counter()
    category_names = {cat['id']: cat['name'] for cat in data['categories']}
    
    for ann in data['annotations']:
        category_counts[ann['category_id']] += 1
    
    # Print category distribution
    table = PrettyTable()
    table.field_names = ["Class ID", "Class Name", "Count", "Percentage"]
    table.align["Class Name"] = "l"
    
    for cat_id, count in sorted(category_counts.items()):
        percentage = (count / num_annotations) * 100
        table.add_row([
            cat_id,
            category_names[cat_id],
            count,
            f"{percentage:.2f}%"
        ])
    
    print("Class Distribution:")
    print(table)
    
    # Image size statistics
    widths = [img['width'] for img in data['images']]
    heights = [img['height'] for img in data['images']]
    
    print(f"\nImage Size Statistics:")
    print(f"Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}")
    print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}")
    
    # Bounding box statistics
    box_widths = []
    box_heights = []
    box_areas = []
    
    for ann in data['annotations']:
        bbox = ann['bbox']
        box_widths.append(bbox[2])
        box_heights.append(bbox[3])
        box_areas.append(bbox[2] * bbox[3])
    
    print(f"\nBounding Box Statistics:")
    print(f"Width - Min: {min(box_widths):.1f}, Max: {max(box_widths):.1f}, Mean: {np.mean(box_widths):.1f}")
    print(f"Height - Min: {min(box_heights):.1f}, Max: {max(box_heights):.1f}, Mean: {np.mean(box_heights):.1f}")
    print(f"Area - Min: {min(box_areas):.1f}, Max: {max(box_areas):.1f}, Mean: {np.mean(box_areas):.1f}")
    
    # Plot distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Class distribution
    categories_list = [category_names[cat_id] for cat_id in sorted(category_counts.keys())]
    counts_list = [category_counts[cat_id] for cat_id in sorted(category_counts.keys())]
    
    axes[0, 0].bar(categories_list, counts_list, color='skyblue')
    axes[0, 0].set_title(f'{split_name} - Class Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Damage Class')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Box size distribution
    axes[0, 1].scatter(box_widths, box_heights, alpha=0.5, s=10)
    axes[0, 1].set_title(f'{split_name} - Bounding Box Sizes', fontweight='bold')
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Height')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box area histogram
    axes[1, 0].hist(box_areas, bins=50, color='green', alpha=0.7)
    axes[1, 0].set_title(f'{split_name} - Box Area Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Area (pixels²)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Annotations per image
    img_ann_counts = Counter()
    for ann in data['annotations']:
        img_ann_counts[ann['image_id']] += 1
    
    ann_per_img = list(img_ann_counts.values())
    axes[1, 1].hist(ann_per_img, bins=20, color='orange', alpha=0.7)
    axes[1, 1].set_title(f'{split_name} - Annotations per Image', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Annotations')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'dataset_analysis_{split_name}.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: dataset_analysis_{split_name}.png")
    
    return data


def main():
    """Main function to analyze all splits"""
    
    # Analyze all splits
    splits = {
        'train': 'CarDD_release/CarDD_COCO/annotations/instances_train2017.json',
        'val': 'CarDD_release/CarDD_COCO/annotations/instances_val2017.json',
        'test': 'CarDD_release/CarDD_COCO/annotations/instances_test2017.json'
    }
    
    for split_name, ann_file in splits.items():
        if Path(ann_file).exists():
            analyze_dataset(ann_file, split_name)
        else:
            print(f"\nWarning: {ann_file} not found!")
    
    print(f"\n{'='*60}")
    print("Dataset analysis complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
