"""
Configuration Management and Utilities
=====================================

Central configuration management and utility functions.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import time


@dataclass
class ProcessingConfig:
    """Main processing configuration."""
    # Quality settings
    quality_preset: str = "high"  # low, medium, high, ultra
    output_format: str = "jpg"
    compression_quality: int = 95
    
    # Performance settings
    max_workers: int = 4
    enable_gpu: bool = False
    memory_limit_mb: int = 2048
    
    # Processing options
    preserve_metadata: bool = True
    auto_enhance: bool = True
    batch_size: int = 8
    
    # Enhancement settings
    enhancement_strength: float = 1.0
    noise_reduction_level: str = "medium"  # low, medium, high
    sharpening_level: str = "medium"
    
    # Output settings
    create_backup: bool = True
    output_suffix: str = "_enhanced"
    
    # Logging
    log_level: str = "INFO"
    save_processing_stats: bool = True


@dataclass 
class ModelConfig:
    """ML model configuration."""
    model_type: str = "esrgan"
    device: str = "auto"
    model_path: Optional[str] = None
    batch_size: int = 1
    precision: str = "fp32"  # fp16, fp32
    enable_tensorrt: bool = False


class ConfigManager:
    """
    Advanced configuration management with validation and persistence.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.processing_config = ProcessingConfig()
        self.model_config = ModelConfig()
        self.custom_configs = {}
        
        # Load default configs
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load configurations from files."""
        # Load processing config
        processing_config_file = self.config_dir / "processing.yaml"
        if processing_config_file.exists():
            self.load_processing_config(processing_config_file)
        
        # Load model config  
        model_config_file = self.config_dir / "models.yaml"
        if model_config_file.exists():
            self.load_model_config(model_config_file)
    
    def load_processing_config(self, config_path: Union[str, Path]) -> None:
        """Load processing configuration from file."""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError("Config file must be YAML or JSON")
        
        # Update processing config
        for key, value in config_data.items():
            if hasattr(self.processing_config, key):
                setattr(self.processing_config, key, value)
    
    def load_model_config(self, config_path: Union[str, Path]) -> None:
        """Load model configuration from file."""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        
        # Update model config
        for key, value in config_data.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
    
    def save_processing_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save processing configuration to file."""
        if config_path is None:
            config_path = self.config_dir / "processing.yaml"
        
        config_path = Path(config_path)
        config_data = asdict(self.processing_config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def save_model_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save model configuration to file."""
        if config_path is None:
            config_path = self.config_dir / "models.yaml"
        
        config_path = Path(config_path)
        config_data = asdict(self.model_config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_quality_settings(self) -> Dict[str, Any]:
        """Get quality-specific settings."""
        quality_presets = {
            "low": {
                "compression_quality": 75,
                "max_workers": 2,
                "batch_size": 4,
                "enhancement_strength": 0.7
            },
            "medium": {
                "compression_quality": 85,
                "max_workers": 4,
                "batch_size": 6,
                "enhancement_strength": 1.0
            },
            "high": {
                "compression_quality": 95,
                "max_workers": 6,
                "batch_size": 8,
                "enhancement_strength": 1.2
            },
            "ultra": {
                "compression_quality": 98,
                "max_workers": 8,
                "batch_size": 4,
                "enhancement_strength": 1.5
            }
        }
        
        return quality_presets.get(self.processing_config.quality_preset, quality_presets["medium"])
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration settings."""
        errors = {}
        
        # Validate processing config
        processing_errors = []
        
        if self.processing_config.quality_preset not in ["low", "medium", "high", "ultra"]:
            processing_errors.append("Invalid quality preset")
        
        if self.processing_config.max_workers < 1 or self.processing_config.max_workers > 16:
            processing_errors.append("max_workers must be between 1 and 16")
        
        if self.processing_config.compression_quality < 1 or self.processing_config.compression_quality > 100:
            processing_errors.append("compression_quality must be between 1 and 100")
        
        if processing_errors:
            errors["processing"] = processing_errors
        
        # Validate model config
        model_errors = []
        
        if self.model_config.device not in ["auto", "cpu", "cuda"]:
            model_errors.append("device must be auto, cpu, or cuda")
        
        if model_errors:
            errors["model"] = model_errors
        
        return errors
    
    def create_preset_configs(self) -> None:
        """Create preset configuration files."""
        # Photography preset
        photography_config = ProcessingConfig(
            quality_preset="high",
            enhancement_strength=1.0,
            noise_reduction_level="medium",
            auto_enhance=True
        )
        
        # Art/Design preset
        art_config = ProcessingConfig(
            quality_preset="ultra",
            enhancement_strength=1.5,
            noise_reduction_level="high",
            auto_enhance=True,
            compression_quality=98
        )
        
        # Performance preset
        performance_config = ProcessingConfig(
            quality_preset="medium",
            max_workers=8,
            batch_size=16,
            enhancement_strength=0.8
        )
        
        # Save presets
        presets_dir = self.config_dir / "presets"
        presets_dir.mkdir(exist_ok=True)
        
        for name, config in [
            ("photography", photography_config),
            ("art", art_config), 
            ("performance", performance_config)
        ]:
            with open(presets_dir / f"{name}.yaml", 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False, indent=2)


class MetricsCalculator:
    """
    Advanced metrics calculation for image quality assessment.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_image_metrics(self, original: 'np.ndarray', processed: 'np.ndarray') -> Dict[str, float]:
        """Calculate comprehensive image quality metrics."""
        import cv2
        import numpy as np
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(original, processed))
        
        # Structural metrics
        metrics.update(self._calculate_structural_metrics(original, processed))
        
        # Perceptual metrics
        metrics.update(self._calculate_perceptual_metrics(original, processed))
        
        # Enhancement metrics
        metrics.update(self._calculate_enhancement_metrics(original, processed))
        
        return metrics
    
    def _calculate_basic_metrics(self, original: 'np.ndarray', processed: 'np.ndarray') -> Dict[str, float]:
        """Calculate basic image metrics."""
        import cv2
        import numpy as np
        
        metrics = {}
        
        # Mean Squared Error
        mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
        metrics['mse'] = float(mse)
        
        # Peak Signal-to-Noise Ratio
        if mse > 0:
            metrics['psnr'] = float(20 * np.log10(255.0 / np.sqrt(mse)))
        else:
            metrics['psnr'] = float('inf')
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(original.astype(np.float64) ** 2)
        noise_power = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
        if noise_power > 0:
            metrics['snr'] = float(10 * np.log10(signal_power / noise_power))
        else:
            metrics['snr'] = float('inf')
        
        return metrics
    
    def _calculate_structural_metrics(self, original: 'np.ndarray', processed: 'np.ndarray') -> Dict[str, float]:
        """Calculate structural similarity metrics."""
        import cv2
        import numpy as np
        
        metrics = {}
        
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            proc_gray = processed
        
        # Structural Similarity Index (SSIM)
        ssim = self._calculate_ssim(orig_gray, proc_gray)
        metrics['ssim'] = ssim
        
        # Edge preservation index
        metrics['edge_preservation'] = self._calculate_edge_preservation(orig_gray, proc_gray)
        
        return metrics
    
    def _calculate_ssim(self, img1: 'np.ndarray', img2: 'np.ndarray') -> float:
        """Calculate Structural Similarity Index."""
        import numpy as np
        
        # Constants for SSIM calculation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        return float(numerator / denominator)
    
    def _calculate_edge_preservation(self, original: 'np.ndarray', processed: 'np.ndarray') -> float:
        """Calculate edge preservation index."""
        import cv2
        import numpy as np
        
        # Calculate edges using Sobel operator
        orig_edges = cv2.Sobel(original, cv2.CV_64F, 1, 1, ksize=3)
        proc_edges = cv2.Sobel(processed, cv2.CV_64F, 1, 1, ksize=3)
        
        # Calculate correlation between edge maps
        correlation = np.corrcoef(orig_edges.flatten(), proc_edges.flatten())[0, 1]
        
        return float(correlation if not np.isnan(correlation) else 0.0)
    
    def _calculate_perceptual_metrics(self, original: 'np.ndarray', processed: 'np.ndarray') -> Dict[str, float]:
        """Calculate perceptual quality metrics."""
        import cv2
        import numpy as np
        
        metrics = {}
        
        # Brightness consistency
        orig_brightness = np.mean(original)
        proc_brightness = np.mean(processed)
        metrics['brightness_change'] = float(proc_brightness - orig_brightness)
        
        # Contrast enhancement ratio
        orig_contrast = np.std(original)
        proc_contrast = np.std(processed)
        metrics['contrast_ratio'] = float(proc_contrast / orig_contrast if orig_contrast > 0 else 1.0)
        
        # Color consistency (for color images)
        if len(original.shape) == 3:
            orig_color_var = np.var(original, axis=2)
            proc_color_var = np.var(processed, axis=2)
            metrics['color_consistency'] = float(np.corrcoef(orig_color_var.flatten(), 
                                                           proc_color_var.flatten())[0, 1])
        
        return metrics
    
    def _calculate_enhancement_metrics(self, original: 'np.ndarray', processed: 'np.ndarray') -> Dict[str, float]:
        """Calculate enhancement-specific metrics."""
        import cv2
        import numpy as np
        
        metrics = {}
        
        # Sharpness improvement
        orig_sharpness = self._calculate_sharpness(original)
        proc_sharpness = self._calculate_sharpness(processed)
        metrics['sharpness_improvement'] = float(proc_sharpness / orig_sharpness if orig_sharpness > 0 else 1.0)
        
        # Noise reduction estimate
        orig_noise = self._estimate_noise(original)
        proc_noise = self._estimate_noise(processed)
        metrics['noise_reduction'] = float((orig_noise - proc_noise) / orig_noise if orig_noise > 0 else 0.0)
        
        return metrics
    
    def _calculate_sharpness(self, image: 'np.ndarray') -> float:
        """Calculate image sharpness using Laplacian variance."""
        import cv2
        import numpy as np
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))
    
    def _estimate_noise(self, image: 'np.ndarray') -> float:
        """Estimate noise level in image."""
        import cv2
        import numpy as np
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use Laplacian to estimate noise
        H, W = gray.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        
        sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray.astype(np.float32), -1, np.array(M)))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
        
        return float(sigma)
    
    def generate_quality_report(self, metrics: Dict[str, float]) -> str:
        """Generate a human-readable quality report."""
        report = "Image Quality Assessment Report\n"
        report += "=" * 35 + "\n\n"
        
        # Basic metrics
        report += "Basic Metrics:\n"
        report += f"  PSNR: {metrics.get('psnr', 0):.2f} dB\n"
        report += f"  SSIM: {metrics.get('ssim', 0):.4f}\n"
        report += f"  MSE: {metrics.get('mse', 0):.2f}\n\n"
        
        # Enhancement metrics  
        report += "Enhancement Analysis:\n"
        report += f"  Sharpness Improvement: {(metrics.get('sharpness_improvement', 1) - 1) * 100:.1f}%\n"
        report += f"  Noise Reduction: {metrics.get('noise_reduction', 0) * 100:.1f}%\n"
        report += f"  Contrast Ratio: {metrics.get('contrast_ratio', 1):.2f}\n"
        report += f"  Brightness Change: {metrics.get('brightness_change', 0):.1f}\n\n"
        
        # Quality assessment
        ssim = metrics.get('ssim', 0)
        if ssim > 0.9:
            quality = "Excellent"
        elif ssim > 0.8:
            quality = "Good"
        elif ssim > 0.6:
            quality = "Fair"
        else:
            quality = "Poor"
        
        report += f"Overall Quality: {quality} (SSIM: {ssim:.4f})\n"
        
        return report


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
        
    def start_timing(self) -> None:
        """Start timing operation."""
        self.start_time = time.time()
    
    def stop_timing(self) -> float:
        """Stop timing and return elapsed time."""
        if self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed
    
    def log_operation(self, operation: str, duration: float, 
                     input_size: tuple, success: bool = True) -> None:
        """Log operation metrics."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append({
            'duration': duration,
            'input_size': input_size,
            'success': success,
            'timestamp': time.time()
        })
    
    def get_average_performance(self, operation: str) -> Dict[str, float]:
        """Get average performance metrics for an operation."""
        if operation not in self.metrics:
            return {}
        
        operations = self.metrics[operation]
        successful_ops = [op for op in operations if op['success']]
        
        if not successful_ops:
            return {}
        
        avg_duration = sum(op['duration'] for op in successful_ops) / len(successful_ops)
        success_rate = len(successful_ops) / len(operations)
        
        return {
            'average_duration': avg_duration,
            'success_rate': success_rate,
            'total_operations': len(operations)
        }
