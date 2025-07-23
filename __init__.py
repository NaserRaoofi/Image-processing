"""
Main package initialization with streamlined imports.
"""

from .src.core import ImageProcessor, BatchProcessor
from .src.enhancement import AdvancedEnhancer, AdaptiveEnhancer
from .src.utils import ConfigManager, MetricsCalculator

__version__ = "2.0.0"
__author__ = "Advanced Computer Vision Team"

# Simplified imports for end users
__all__ = [
    'ImageProcessor',
    'BatchProcessor', 
    'AdvancedEnhancer',
    'AdaptiveEnhancer',
    'ConfigManager',
    'MetricsCalculator'
]
