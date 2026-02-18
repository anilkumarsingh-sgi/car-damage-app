"""
Data augmentation and transformation pipelines for CarDD dataset
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(img_size: int = 640, use_advanced: bool = True):
    """
    Get training data augmentation pipeline.
    
    Args:
        img_size: Target image size
        use_advanced: Whether to use advanced augmentations
    
    Returns:
        Albumentations Compose object
    """
    if use_advanced:
        transforms = [
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            # Perspective and distortion
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            ], p=0.3),
            
            # Color transformations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            A.OneOf([
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
                A.ChannelShuffle(p=1.0),
                A.ColorJitter(p=1.0),
            ], p=0.3),
            
            # Blur and noise
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0),
            ], p=0.2),
            
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Weather effects
            A.OneOf([
                A.RandomRain(slant_lower=-10, slant_upper=10, p=1.0),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, p=1.0),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1.0),
            ], p=0.2),
            
            # Image quality
            A.OneOf([
                A.ImageCompression(quality_lower=75, quality_upper=100, p=1.0),
                A.Downscale(scale_min=0.75, scale_max=0.95, p=1.0),
            ], p=0.2),
            
            # CLAHE for contrast
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            
            # Cutout/CoarseDropout for robustness
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
            
            # Resize and normalize
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    else:
        # Basic transformations
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=0,
            min_visibility=0.3
        )
    )


def get_val_transforms(img_size: int = 640):
    """
    Get validation/test data transformation pipeline.
    
    Args:
        img_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    transforms = [
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=0,
            min_visibility=0
        )
    )


def get_sod_train_transforms(img_size: int = 320):
    """
    Get training transforms for SOD (Salient Object Detection) format.
    
    Args:
        img_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
    
    return A.Compose(transforms)


def get_sod_val_transforms(img_size: int = 320):
    """
    Get validation transforms for SOD format.
    
    Args:
        img_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    transforms = [
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
    
    return A.Compose(transforms)


def get_tta_transforms(img_size: int = 640):
    """
    Get Test Time Augmentation (TTA) transforms.
    
    Args:
        img_size: Target image size
    
    Returns:
        List of Albumentations Compose objects
    """
    tta_transforms = [
        # Original
        A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Vertical flip
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Multi-scale
        A.Compose([
            A.Resize(int(img_size * 1.2), int(img_size * 1.2)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    
    return tta_transforms
