#!/usr/bin/env python3
"""
Advanced Image Processing Demo
=============================

Demonstration of the enhanced image processing capabilities with ML/DL integration.
This script showcases the transformation from basic image processing to enterprise-grade toolkit.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_image():
    """Create a demo image for testing if no images are available."""
    # Create a synthetic image with various challenges
    height, width = 400, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        for j in range(width):
            image[i, j] = [int(i * 255 / height), int(j * 255 / width), 128]
    
    # Add some geometric shapes
    cv2.circle(image, (150, 150), 50, (255, 255, 255), -1)
    cv2.rectangle(image, (400, 100), (550, 250), (0, 255, 255), -1)
    
    # Add noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Make it darker to simulate low-light
    image = (image * 0.4).astype(np.uint8)
    
    return image


def demonstrate_traditional_vs_enhanced():
    """Demonstrate the difference between traditional and enhanced processing."""
    print("\n🎯 TRADITIONAL vs ENHANCED PROCESSING COMPARISON")
    print("=" * 60)
    
    # Try to use an existing image, otherwise create demo
    image_path = Path("Top Image/Messi.jpg")
    if image_path.exists():
        image = cv2.imread(str(image_path))
        print(f"✅ Using existing image: {image_path}")
    else:
        image = create_demo_image()
        print("✅ Using synthetic demo image")
    
    # Original processing (from the old codebase)
    print("\n📊 Traditional Processing:")
    start_time = time.time()
    
    # Traditional CLAHE
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        traditional_result = cv2.merge([l, a, b])
        traditional_result = cv2.cvtColor(traditional_result, cv2.COLOR_LAB2BGR)
    
    traditional_time = time.time() - start_time
    print(f"   ⏱️  Processing time: {traditional_time:.3f}s")
    print(f"   🔧 Method: Basic CLAHE")
    
    # Enhanced processing (new capabilities)
    print("\n🚀 Enhanced Processing:")
    start_time = time.time()
    
    try:
        from src.enhancement import AdaptiveEnhancer
        from src.utils import MetricsCalculator
        
        enhancer = AdaptiveEnhancer()
        metrics_calc = MetricsCalculator()
        
        # Intelligent auto-enhancement
        enhanced_result = enhancer.auto_enhance(image)
        
        enhanced_time = time.time() - start_time
        print(f"   ⏱️  Processing time: {enhanced_time:.3f}s")
        print(f"   🔧 Method: AI-powered adaptive enhancement")
        
        # Calculate quality metrics
        if traditional_result is not None:
            metrics = metrics_calc.calculate_image_metrics(image, enhanced_result)
            print(f"   📈 SSIM: {metrics.get('ssim', 0):.4f}")
            print(f"   📈 PSNR: {metrics.get('psnr', 0):.2f} dB")
            print(f"   📈 Contrast Improvement: {((metrics.get('contrast_ratio', 1) - 1) * 100):.1f}%")
        
        # Save results for comparison
        cv2.imwrite("demo_traditional.jpg", traditional_result)
        cv2.imwrite("demo_enhanced.jpg", enhanced_result)
        print(f"   💾 Results saved: demo_traditional.jpg, demo_enhanced.jpg")
        
    except ImportError as e:
        print(f"   ❌ Enhanced processing not available: {e}")
        print(f"   💡 Install requirements: pip install -r requirements.txt")


def demonstrate_ml_capabilities():
    """Demonstrate ML/DL capabilities if available."""
    print("\n🤖 MACHINE LEARNING CAPABILITIES")
    print("=" * 40)
    
    try:
        # Check if PyTorch is available
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        from src.ml_models import DeepEnhancer, ModelManager
        
        # Model management demo
        model_manager = ModelManager()
        
        # Register models
        esrgan = DeepEnhancer("esrgan")
        model_manager.register_model("esrgan", esrgan)
        
        print("✅ ML Models registered:")
        for model in model_manager.list_models():
            info = model_manager.get_model_info(model)
            print(f"   📋 {model}: {info['type']}")
        
        print("\n💡 Available ML Features:")
        print("   🔍 Super-resolution with ESRGAN")
        print("   🎨 Style transfer")
        print("   🔇 Advanced denoising with DnCNN")
        print("   🌙 Low-light enhancement with EnlightenGAN")
        
    except ImportError:
        print("❌ PyTorch not available")
        print("💡 Install ML dependencies: pip install torch torchvision")
        print("💡 Available features limited to traditional processing")


def demonstrate_api_capabilities():
    """Demonstrate API and CLI capabilities."""
    print("\n🌐 API & CLI CAPABILITIES")
    print("=" * 30)
    
    print("✅ Available Interfaces:")
    print("   🐍 Python API - Direct library usage")
    print("   💻 CLI Tool - Command-line interface")
    print("   🌐 REST API - Web service with FastAPI")
    print("   📊 Batch Processing - Parallel processing")
    
    print("\n📋 CLI Examples:")
    print("   image-process enhance input.jpg --method auto")
    print("   image-process batch ./images --output ./processed")
    print("   image-process analyze original.jpg enhanced.jpg")
    
    print("\n🌐 API Examples:")
    print("   POST /enhance - Single image enhancement")
    print("   POST /batch - Batch processing")
    print("   POST /analyze - Quality analysis")
    print("   GET /health - Health check")


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("\n⚙️  CONFIGURATION MANAGEMENT")
    print("=" * 35)
    
    try:
        from src.utils import ConfigManager
        
        config_manager = ConfigManager()
        
        print("✅ Configuration Features:")
        print("   📝 YAML/JSON configuration files")
        print("   🎛️  Quality presets (low, medium, high, ultra)")
        print("   🔧 Custom algorithm parameters")
        print("   🚀 Performance tuning options")
        
        # Show current config
        config_dict = {
            'quality_preset': config_manager.processing_config.quality_preset,
            'max_workers': config_manager.processing_config.max_workers,
            'enable_gpu': config_manager.processing_config.enable_gpu,
            'enhancement_strength': config_manager.processing_config.enhancement_strength
        }
        
        print("\n📊 Current Configuration:")
        for key, value in config_dict.items():
            print(f"   {key}: {value}")
        
        # Create preset configs
        config_manager.create_preset_configs()
        print("✅ Preset configurations created in config/presets/")
        
    except ImportError:
        print("❌ Configuration management not available")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n📊 PERFORMANCE MONITORING")
    print("=" * 30)
    
    try:
        from src.utils import PerformanceMonitor, MetricsCalculator
        from src.core import ImageProcessor
        
        monitor = PerformanceMonitor()
        processor = ImageProcessor()
        
        # Create test image
        test_image = create_demo_image()
        
        print("✅ Performance Monitoring Features:")
        print("   ⏱️  Processing time tracking")
        print("   📈 Quality metrics calculation")
        print("   📊 Throughput measurement")
        print("   💾 Statistics persistence")
        
        # Benchmark different operations
        operations = ['enhance', 'denoise', 'sharpen']
        results = {}
        
        print(f"\n🏃 Benchmarking {len(operations)} operations...")
        
        for operation in operations:
            monitor.start_timing()
            processed = processor.process(test_image, operation)
            duration = monitor.stop_timing()
            monitor.log_operation(operation, duration, test_image.shape)
            results[operation] = duration
            print(f"   {operation}: {duration:.3f}s")
        
        # Show statistics
        stats = processor.get_stats()
        print(f"\n📊 Processing Statistics:")
        print(f"   Total processed: {stats['total_processed']}")
        print(f"   Average time: {stats.get('avg_time_per_image', 0):.3f}s")
        print(f"   Error rate: {stats.get('error_rate', 0):.1%}")
        
    except ImportError as e:
        print(f"❌ Performance monitoring not available: {e}")


def main():
    """Main demo function."""
    print("🚀 ADVANCED IMAGE PROCESSING TOOLKIT DEMO")
    print("=" * 50)
    print("Showcasing enterprise-grade image processing capabilities")
    print("Enhanced by 20+ years of ML engineering experience\n")
    
    # Create output directory
    Path("demo_output").mkdir(exist_ok=True)
    
    # Run demonstrations
    demonstrate_traditional_vs_enhanced()
    demonstrate_ml_capabilities()
    demonstrate_api_capabilities() 
    demonstrate_configuration()
    demonstrate_performance_monitoring()
    
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETE!")
    print("=" * 50)
    print("\n📚 Next Steps:")
    print("1. 📖 Read the documentation: README.md")
    print("2. 🛠️  Install dependencies: pip install -r requirements.txt")
    print("3. 🚀 Start the API: python api.py")
    print("4. 💻 Try the CLI: python cli.py --help")
    print("5. 🔧 Customize config: config/processing.yaml")
    
    print("\n💡 Key Improvements:")
    print("✅ Modular, enterprise-grade architecture")
    print("✅ ML/DL integration with PyTorch")
    print("✅ Production-ready REST API")
    print("✅ Comprehensive CLI interface")
    print("✅ Advanced quality metrics")
    print("✅ Performance monitoring")
    print("✅ Flexible configuration management")
    print("✅ Docker & Kubernetes ready")
    
    print("\n🌟 From basic image processing to production-ready CV toolkit!")


if __name__ == "__main__":
    main()
