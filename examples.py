#!/usr/bin/env python3
"""
Comprehensive Usage Examples
============================

This script demonstrates how to use the enhanced image processing toolkit
in various scenarios, from basic usage to advanced enterprise features.
"""

def example_basic_usage():
    """Basic usage examplperformance_config = CoreProcessingConfig(
    # Optimized for speed
    clahe_clip_limit=2.0,
    clahe_grid_size=(6, 6),
    gaussian_blur_sigma=0.5,
    bilateral_sigma_color=30,
    bilateral_sigma_space=30
)acing the old scripts."""
    print("📝 BASIC USAGE EXAMPLE")
    print("=" * 25)
    
    # Old way (from original scripts)
    print("🔴 Old way (HSV_CLAHE_LAB.py equivalent):")
    print("""
    import cv2
    import numpy as np
    
    # Load image
    image = cv2.imread("input.jpg")
    
    # Manual CLAHE processing
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    v = clahe.apply(v)
    result = cv2.merge([h, s, v])
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    """)
    
    print("\n🟢 New way (Enhanced toolkit):")
    print("""
    from src.core import ImageProcessor
    from src.enhancement import AdaptiveEnhancer
    import cv2
    
    # Load image
    image = cv2.imread("input.jpg")
    
    # One-line intelligent enhancement
    enhancer = AdaptiveEnhancer()
    result = enhancer.auto_enhance(image)
    
    # Or specific enhancement
    processor = ImageProcessor()
    result = processor.process(image, 'enhance', method='clahe')
    """)


def example_advanced_features():
    """Advanced features example."""
    print("\n🚀 ADVANCED FEATURES EXAMPLE")
    print("=" * 30)
    
    print("""
# Multi-scale enhancement with quality metrics
from src.enhancement import AdvancedEnhancer
from src.utils import MetricsCalculator
import cv2

# Load image
image = cv2.imread("input.jpg")

# Advanced enhancement
enhancer = AdvancedEnhancer()

# Low-light enhancement with multiple algorithms
retinex_result = enhancer.enhance_low_light(image, method='retinex')
lime_result = enhancer.enhance_low_light(image, method='lime')

# Color enhancement with skin tone protection
vibrant_result = enhancer.enhance_color(image, method='vibrance')

# Quality assessment
metrics_calc = MetricsCalculator()
quality_metrics = metrics_calc.calculate_image_metrics(image, retinex_result)

print(f"SSIM: {quality_metrics['ssim']:.4f}")
print(f"PSNR: {quality_metrics['psnr']:.2f} dB")

# Generate quality report
report = metrics_calc.generate_quality_report(quality_metrics)
print(report)
""")


def example_batch_processing():
    """Batch processing example."""
    print("\n📦 BATCH PROCESSING EXAMPLE")
    print("=" * 27)
    
    print("""
# High-performance batch processing
from src.core import ImageProcessor, BatchProcessor
from pathlib import Path

# Setup
processor = ImageProcessor()
batch_processor = BatchProcessor(processor, max_workers=8)

# Process entire directory
results = batch_processor.process_directory(
    input_dir="./input_images",
    output_dir="./enhanced_images", 
    operation="enhance"
)

print(f"Processed: {results['processed']} images")
print(f"Failed: {results['failed']} images")
print(f"Total time: {results['total_time']:.2f}s")

# Custom processing with progress callback
def progress_callback(current, total):
    print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

# Process with custom parameters
for img_file in Path("./images").glob("*.jpg"):
    image = cv2.imread(str(img_file))
    
    # Adaptive enhancement based on image characteristics
    enhanced = processor.process(image, 'enhance', method='adaptive')
    
    # Save with quality metrics
    output_path = Path("./output") / f"enhanced_{img_file.name}"
    cv2.imwrite(str(output_path), enhanced)
""")


def example_ml_integration():
    """ML/DL integration example."""
    print("\n🤖 ML/DL INTEGRATION EXAMPLE")
    print("=" * 29)
    
    print("""
# Deep learning model integration
from src.ml_models import DeepEnhancer, StyleTransfer, ModelManager
import cv2

# Setup model manager
model_manager = ModelManager()

# Register and load ESRGAN for super-resolution
esrgan = DeepEnhancer("esrgan", device="cuda")  # or "cpu"
model_manager.register_model("esrgan", esrgan)
model_manager.load_model("esrgan", "models/esrgan_weights.pth")

# Super-resolution enhancement
low_res_image = cv2.imread("low_res.jpg")
high_res_result = model_manager.predict("esrgan", low_res_image)

# Style transfer
style_transfer = StyleTransfer(device="cuda")
model_manager.register_model("style_transfer", style_transfer)
model_manager.load_model("style_transfer", "models/style_model.pth")

content_image = cv2.imread("content.jpg")
stylized_result = style_transfer.transfer_style(content_image, style_strength=0.8)

# List available models
print("Available models:", model_manager.list_models())

# Model information
for model_name in model_manager.list_models():
    info = model_manager.get_model_info(model_name)
    print(f"{model_name}: {info}")
""")


def example_api_usage():
    """API usage examples."""
    print("\n🌐 API USAGE EXAMPLES")
    print("=" * 21)
    
    print("📡 REST API Usage:")
    print("""
# Start the API server
python api.py

# Or with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Python requests example
import requests

# Single image enhancement
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/enhance",
        files={"file": f},
        data={
            "method": "auto",
            "strength": 1.2,
            "return_base64": True
        }
    )

result = response.json()
print(f"Processing time: {result['processing_time']:.3f}s")
print(f"Quality metrics: {result['metrics']}")

# Batch processing
files = [
    ("files", ("img1.jpg", open("img1.jpg", "rb"), "image/jpeg")),
    ("files", ("img2.jpg", open("img2.jpg", "rb"), "image/jpeg")),
]

response = requests.post(
    "http://localhost:8000/batch",
    files=files,
    data={"operation": "enhance", "format": "jpg"}
)

job_info = response.json()
job_id = job_info["job_id"]

# Check job status
status_response = requests.get(f"http://localhost:8000/job/{job_id}")
print(status_response.json())
""")
    
    print("\n💻 CLI Usage:")
    print("""
# Basic enhancement
python cli.py enhance input.jpg --output enhanced.jpg --method auto

# Low-light enhancement
python cli.py enhance dark_image.jpg --method low_light --algorithm retinex

# Batch processing with custom parameters
python cli.py batch ./images --output ./processed --operation enhance --workers 8

# ML-based super-resolution
python cli.py ml-enhance low_res.jpg --model esrgan --scale 2 --device cuda

# Quality analysis with detailed report
python cli.py analyze original.jpg enhanced.jpg --report quality_report.json

# Configuration management
python cli.py config show
python cli.py config set quality_preset ultra
python cli.py config create-presets

# Model management
python cli.py models list
python cli.py models load esrgan --path ./models/esrgan.pth
python cli.py models info esrgan
""")


def example_configuration():
    """Configuration management example."""
    print("\n⚙️ CONFIGURATION EXAMPLE")
    print("=" * 25)
    
    print("""
# Advanced configuration management
from src.utils import ConfigManager
from src.core import CoreProcessingConfig

# Load configuration manager
config_manager = ConfigManager()

# Customize processing configuration
config_manager.processing_config.quality_preset = "ultra"
config_manager.processing_config.enhancement_strength = 1.5
config_manager.processing_config.max_workers = 8
config_manager.processing_config.enable_gpu = True

# Save configuration
config_manager.save_processing_config("my_config.yaml")

# Load custom configuration
config_manager.load_processing_config("photography_preset.yaml")

# Get quality-specific settings
quality_settings = config_manager.get_quality_settings()
print(quality_settings)

# Validate configuration
errors = config_manager.validate_config()
if errors:
    print("Configuration errors:", errors)

# Create preset configurations
config_manager.create_preset_configs()

# Custom configuration for specific use cases
photography_config = CoreProcessingConfig(
    # Optimized for high-quality photography
    clahe_clip_limit=3.0,
    clahe_grid_size=(12, 12),
    gaussian_blur_sigma=0.8,
    bilateral_sigma_color=80,
    bilateral_sigma_space=80
)

# Performance-oriented configuration
performance_config = ProcessingConfig(
    quality_preset="medium", 
    max_workers=16,
    batch_size=32,
    enable_gpu=True,
    memory_limit_mb=4096
)
""")


def example_performance_monitoring():
    """Performance monitoring example."""
    print("\n📊 PERFORMANCE MONITORING EXAMPLE")
    print("=" * 34)
    
    print("""
# Comprehensive performance monitoring
from src.utils import PerformanceMonitor, MetricsCalculator
from src.core import ImageProcessor
import cv2

# Setup monitoring
monitor = PerformanceMonitor()
processor = ImageProcessor()
metrics_calc = MetricsCalculator()

# Process with timing
monitor.start_timing()
image = cv2.imread("input.jpg")
enhanced = processor.process(image, "enhance")
duration = monitor.stop_timing()

# Log operation
monitor.log_operation("enhance", duration, image.shape, success=True)

# Calculate quality metrics
quality_metrics = metrics_calc.calculate_image_metrics(image, enhanced)

# Get performance statistics
performance_stats = monitor.get_average_performance("enhance")
print(f"Average processing time: {performance_stats['average_duration']:.3f}s")
print(f"Success rate: {performance_stats['success_rate']:.1%}")

# Processor statistics
processor_stats = processor.get_stats()
print(f"Total processed: {processor_stats['total_processed']}")
print(f"Average time per image: {processor_stats.get('avg_time_per_image', 0):.3f}s")
print(f"Error rate: {processor_stats.get('error_rate', 0):.1%}")

# Generate comprehensive quality report
quality_report = metrics_calc.generate_quality_report(quality_metrics)
print(quality_report)

# Performance benchmarking
def benchmark_operations():
    operations = ['enhance', 'denoise', 'sharpen', 'blur']
    results = {}
    
    for op in operations:
        monitor.start_timing()
        result = processor.process(image, op)
        time_taken = monitor.stop_timing()
        results[op] = time_taken
        
    return results

benchmark_results = benchmark_operations()
for op, time_taken in benchmark_results.items():
    print(f"{op}: {time_taken:.3f}s")
""")


def example_enterprise_deployment():
    """Enterprise deployment example."""
    print("\n🏭 ENTERPRISE DEPLOYMENT EXAMPLE")
    print("=" * 32)
    
    print("""
# Docker deployment
docker build -t advanced-image-processing .
docker run -p 8000:8000 -e ENABLE_GPU=true advanced-image-processing

# Docker Compose with Redis and monitoring
docker-compose up -d

# Kubernetes deployment
kubectl apply -f k8s/

# Environment configuration
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Production configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-processing-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: image-processing
  template:
    spec:
      containers:
      - name: api
        image: advanced-image-processing:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
        env:
        - name: ENABLE_GPU
          value: "true"
        - name: MAX_WORKERS
          value: "8"

# Load balancing and autoscaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: image-processing-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: image-processing-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
""")


def main():
    """Main function to run all examples."""
    print("📚 COMPREHENSIVE USAGE EXAMPLES")
    print("=" * 35)
    print("Advanced Image Processing Toolkit - Usage Guide\n")
    
    example_basic_usage()
    example_advanced_features() 
    example_batch_processing()
    example_ml_integration()
    example_api_usage()
    example_configuration()
    example_performance_monitoring()
    example_enterprise_deployment()
    
    print("\n" + "=" * 50)
    print("🎓 EXAMPLES COMPLETE!")
    print("=" * 50)
    print("\n🚀 Ready to transform your image processing workflow!")
    print("📖 See README.md for complete documentation")
    print("🔧 Run demo.py for live demonstrations")


if __name__ == "__main__":
    main()
