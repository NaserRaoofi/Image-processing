"""
Advanced Image Processing & Computer Vision Toolkit
==================================================

A comprehensive, production-ready computer vision library with modern ML/DL capabilities.

Author: Enhanced by Senior ML Engineer
Version: 2.0.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Advanced Computer Vision Team"

from .core import ImageProcessor, BatchProcessor
from .enhancement import AdvancedEnhancer, AdaptiveEnhancer
from .detection import ObjectDetector, FaceDetector
from .segmentation import SemanticSegmenter, InstanceSegmenter
from .filters import NoiseReducer, EdgeDetector
from .transforms import GeometricTransforms, ColorTransforms
from .analysis import ImageAnalyzer, QualityAssessment
from .ml_models import DeepEnhancer, StyleTransfer
from .utils import ConfigManager, MetricsCalculator

__all__ = [
    'ImageProcessor',
    'BatchProcessor', 
    'AdvancedEnhancer',
    'AdaptiveEnhancer',
    'ObjectDetector',
    'FaceDetector',
    'SemanticSegmenter',
    'InstanceSegmenter',
    'NoiseReducer',
    'EdgeDetector',
    'GeometricTransforms',
    'ColorTransforms',
    'ImageAnalyzer',
    'QualityAssessment',
    'DeepEnhancer',
    'StyleTransfer',
    'ConfigManager',
    'MetricsCalculator'
]
