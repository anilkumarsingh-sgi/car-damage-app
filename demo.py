#!/usr/bin/env python
"""
Demo script showcasing CarDD model capabilities
"""

import cv2
import numpy as np
from pathlib import Path
import torch

# Assuming the model is already set up
from inference import CarDDInference


def demo_single_image():
    """Demo: Single image inference"""
    print("\n" + "="*60)
    print("DEMO 1: Single Image Inference")
    print("="*60)
    
    # Initialize model
    model = CarDDInference(
        model_path='checkpoints/best_model.pth',
        config_path='config/config.yaml',
        device='cuda'
    )
    
    # Run inference
    prediction = model.predict_image(
        'examples/car_damage_1.jpg',
        conf_threshold=0.25,
        save_path='outputs/demo_result.jpg',
        visualize=True
    )
    
    # Print results
    print(f"\nDetected {len(prediction['boxes'])} damage(s):")
    for i, (box, label, score) in enumerate(zip(
        prediction['boxes'],
        prediction['labels'],
        prediction['scores']
    )):
        class_name = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat'][label]
        print(f"  {i+1}. {class_name}: {score:.2%} confidence")
    
    print("\nResult saved to: outputs/demo_result.jpg")


def demo_batch_processing():
    """Demo: Batch image processing"""
    print("\n" + "="*60)
    print("DEMO 2: Batch Image Processing")
    print("="*60)
    
    model = CarDDInference(
        model_path='checkpoints/best_model.pth',
        config_path='config/config.yaml'
    )
    
    # Process all images in a directory
    model.predict_batch(
        image_dir='examples/test_images/',
        output_dir='outputs/batch_results/',
        conf_threshold=0.25
    )
    
    print("\nBatch processing complete! Check outputs/batch_results/")


def demo_video_inference():
    """Demo: Video inference"""
    print("\n" + "="*60)
    print("DEMO 3: Video Inference")
    print("="*60)
    
    model = CarDDInference(
        model_path='checkpoints/best_model.pth',
        config_path='config/config.yaml'
    )
    
    # Process video
    model.predict_video(
        video_path='examples/car_damage_video.mp4',
        output_path='outputs/output_video.mp4',
        conf_threshold=0.25,
        show_preview=True
    )
    
    print("\nVideo processing complete! Saved to: outputs/output_video.mp4")


def demo_custom_visualization():
    """Demo: Custom visualization"""
    print("\n" + "="*60)
    print("DEMO 4: Custom Visualization")
    print("="*60)
    
    from src.utils.visualization import visualize_predictions, create_comparison_plot
    
    # Load image
    image = cv2.imread('examples/car_damage_1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Mock prediction (for demo)
    prediction = {
        'boxes': torch.tensor([[100, 150, 300, 400], [350, 200, 500, 450]]),
        'labels': torch.tensor([0, 2]),  # dent and crack
        'scores': torch.tensor([0.92, 0.87])
    }
    
    # Visualize
    result = visualize_predictions(
        image,
        prediction,
        conf_threshold=0.5,
        show_labels=True,
        show_conf=True,
        line_thickness=3,
        font_scale=0.8
    )
    
    # Save
    cv2.imwrite('outputs/custom_viz.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print("\nCustom visualization saved to: outputs/custom_viz.jpg")


def demo_model_comparison():
    """Demo: Compare different model architectures"""
    print("\n" + "="*60)
    print("DEMO 5: Model Architecture Comparison")
    print("="*60)
    
    models = {
        'YOLOv8n': 'checkpoints/yolov8n_best.pth',
        'YOLOv8x': 'checkpoints/yolov8x_best.pth',
        'Mask R-CNN': 'checkpoints/maskrcnn_best.pth',
        'Hybrid': 'checkpoints/hybrid_best.pth'
    }
    
    test_image = 'examples/car_damage_1.jpg'
    
    import time
    from prettytable import PrettyTable
    
    table = PrettyTable()
    table.field_names = ["Model", "Detections", "Avg Confidence", "Inference Time (ms)"]
    
    for model_name, model_path in models.items():
        if not Path(model_path).exists():
            print(f"Skipping {model_name} (checkpoint not found)")
            continue
        
        # Initialize model
        model = CarDDInference(model_path, 'config/config.yaml')
        
        # Measure inference time
        start_time = time.time()
        prediction = model.predict_image(test_image, visualize=False)
        inference_time = (time.time() - start_time) * 1000
        
        # Calculate stats
        num_detections = len(prediction['boxes'])
        avg_conf = prediction['scores'].mean().item() if num_detections > 0 else 0
        
        table.add_row([
            model_name,
            num_detections,
            f"{avg_conf:.2%}",
            f"{inference_time:.1f}"
        ])
    
    print("\n" + str(table))


def demo_ensemble_prediction():
    """Demo: Ensemble multiple models for better accuracy"""
    print("\n" + "="*60)
    print("DEMO 6: Ensemble Prediction")
    print("="*60)
    
    # Load multiple models
    model1 = CarDDInference('checkpoints/yolov8x_best.pth', 'config/config.yaml')
    model2 = CarDDInference('checkpoints/maskrcnn_best.pth', 'config/config.yaml')
    
    # Get predictions from both
    image = 'examples/car_damage_1.jpg'
    pred1 = model1.predict_image(image, visualize=False)
    pred2 = model2.predict_image(image, visualize=False)
    
    # Simple ensemble: combine boxes with NMS
    # (In practice, use more sophisticated fusion)
    all_boxes = torch.cat([pred1['boxes'], pred2['boxes']])
    all_scores = torch.cat([pred1['scores'], pred2['scores']])
    all_labels = torch.cat([pred1['labels'], pred2['labels']])
    
    print(f"\nModel 1 detections: {len(pred1['boxes'])}")
    print(f"Model 2 detections: {len(pred2['boxes'])}")
    print(f"Combined detections: {len(all_boxes)}")
    print("\nEnsemble can improve robustness and accuracy!")


def main():
    """Run all demos"""
    
    print("\n" + "="*70)
    print(" "*20 + "CarDD Model Demos")
    print("="*70)
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    Path('examples').mkdir(exist_ok=True)
    
    demos = [
        ("Single Image Inference", demo_single_image),
        ("Batch Processing", demo_batch_processing),
        ("Video Inference", demo_video_inference),
        ("Custom Visualization", demo_custom_visualization),
        ("Model Comparison", demo_model_comparison),
        ("Ensemble Prediction", demo_ensemble_prediction)
    ]
    
    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\nRun individual demos or all at once.")
    print("\nNote: Make sure you have trained models in checkpoints/ directory")
    print("      and sample images in examples/ directory\n")
    
    # Uncomment to run all demos:
    # for name, demo_func in demos:
    #     try:
    #         demo_func()
    #     except Exception as e:
    #         print(f"\nError in {name}: {str(e)}")
    #         print("Continuing to next demo...\n")


if __name__ == '__main__':
    main()
