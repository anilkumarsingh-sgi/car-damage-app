"""
Export CarDD models to various formats
"""

import argparse
import torch
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent))

from src.models.yolo_model import create_yolo_model
from src.models.maskrcnn_model import create_maskrcnn_model
from src.models.hybrid_model import create_hybrid_model
from src.utils.logger import setup_logger


def export_model(
    model_path: str,
    config_path: str,
    output_dir: str,
    format: str = 'onnx',
    img_size: int = 640,
    device: str = 'cuda'
):
    """
    Export model to specified format.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        output_dir: Output directory
        format: Export format (onnx, torchscript, tflite)
        img_size: Input image size
        device: Device to run export on
    """
    logger = setup_logger('CarDD_Export')
    logger.info(f"Exporting model to {format}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_arch = config['model']['architecture']
    num_classes = config['dataset']['num_classes']
    
    if model_arch == 'yolov8':
        model = create_yolo_model(
            model_size=config['model']['backbone'].replace('yolov8', ''),
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load weights
        model.load_weights(model_path)
        
        # Export using YOLO's built-in export
        export_path = model.export(format=format, dynamic=True, simplify=True)
        logger.info(f"Model exported to: {export_path}")
        
    elif model_arch in ['mask_rcnn', 'hybrid']:
        if model_arch == 'mask_rcnn':
            model = create_maskrcnn_model(num_classes=num_classes + 1, pretrained=False)
        else:
            model = create_hybrid_model(num_classes=num_classes, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # Dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        
        if format == 'onnx':
            output_path = output_dir / f'{model_arch}.onnx'
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"ONNX model exported to: {output_path}")
            
        elif format == 'torchscript':
            output_path = output_dir / f'{model_arch}.pt'
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(output_path))
            logger.info(f"TorchScript model exported to: {output_path}")
            
        else:
            logger.error(f"Unsupported export format: {format}")
    
    logger.info("Export completed!")


def main():
    parser = argparse.ArgumentParser(description='Export CarDD Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='exports',
                        help='Output directory')
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'tflite', 'coreml'],
                        help='Export format')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run export on')
    
    args = parser.parse_args()
    
    export_model(
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output,
        format=args.format,
        img_size=args.img_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
