"""
Advanced CLI Interface for Image Processing
==========================================

Enterprise-grade command-line interface with comprehensive features.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import time

# Import our modules
from src.core import ImageProcessor, BatchProcessor, CoreProcessingConfig
from src.enhancement import AdvancedEnhancer, AdaptiveEnhancer
from src.ml_models import DeepEnhancer, StyleTransfer, ModelManager
from src.utils import ConfigManager, MetricsCalculator, PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageProcessingCLI:
    """
    Advanced CLI for image processing with enterprise features.
    """
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.metrics_calculator = MetricsCalculator()
        self.performance_monitor = PerformanceMonitor()
        self.model_manager = ModelManager()
        
        # Initialize processors
        self.image_processor = ImageProcessor(self.config_manager.processing_config)
        self.batch_processor = BatchProcessor(self.image_processor)
        self.advanced_enhancer = AdvancedEnhancer()
        self.adaptive_enhancer = AdaptiveEnhancer()
        
        # Setup ML models
        self._setup_ml_models()
    
    def _setup_ml_models(self) -> None:
        """Setup ML models if available."""
        try:
            # Register deep learning models
            deep_enhancer = DeepEnhancer("esrgan")
            self.model_manager.register_model("esrgan", deep_enhancer)
            
            style_transfer = StyleTransfer()
            self.model_manager.register_model("style_transfer", style_transfer)
            
            logger.info("ML models registered successfully")
        except Exception as e:
            logger.warning(f"ML models not available: {e}")
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser."""
        parser = argparse.ArgumentParser(
            description="Advanced Image Processing CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic enhancement
  python cli.py enhance input.jpg --output enhanced.jpg
  
  # Batch processing
  python cli.py batch ./images --output ./processed --operation enhance
  
  # Low-light enhancement
  python cli.py enhance dark_image.jpg --method low_light --algorithm retinex
  
  # Super-resolution with ML
  python cli.py ml-enhance input.jpg --model esrgan --scale 2
  
  # Style transfer
  python cli.py style-transfer content.jpg --style-strength 0.8
  
  # Quality assessment
  python cli.py analyze original.jpg enhanced.jpg
            """
        )
        
        # Global options
        parser.add_argument('--config', type=str, help='Configuration file path')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
        parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Enhance command
        self._add_enhance_parser(subparsers)
        
        # Batch processing command
        self._add_batch_parser(subparsers)
        
        # ML enhancement command
        self._add_ml_enhance_parser(subparsers)
        
        # Style transfer command
        self._add_style_transfer_parser(subparsers)
        
        # Analysis command
        self._add_analyze_parser(subparsers)
        
        # Configuration commands
        self._add_config_parser(subparsers)
        
        # Model management commands
        self._add_model_parser(subparsers)
        
        return parser
    
    def _add_enhance_parser(self, subparsers) -> None:
        """Add enhancement command parser."""
        enhance_parser = subparsers.add_parser('enhance', help='Enhance single image')
        enhance_parser.add_argument('input', help='Input image path')
        enhance_parser.add_argument('--output', '-o', help='Output image path')
        enhance_parser.add_argument('--method', choices=[
            'auto', 'clahe', 'adaptive', 'low_light', 'contrast', 'color', 'sharpen', 'denoise'
        ], default='auto', help='Enhancement method')
        enhance_parser.add_argument('--algorithm', choices=[
            'retinex', 'lime', 'bilateral', 'nlmeans', 'vibrance', 'saturation'
        ], help='Specific algorithm for the method')
        enhance_parser.add_argument('--strength', type=float, default=1.0, 
                                  help='Enhancement strength (0.1-2.0)')
        enhance_parser.add_argument('--preserve-colors', action='store_true',
                                  help='Preserve original colors')
        enhance_parser.add_argument('--no-backup', action='store_true',
                                  help='Do not create backup of original')
    
    def _add_batch_parser(self, subparsers) -> None:
        """Add batch processing command parser."""
        batch_parser = subparsers.add_parser('batch', help='Batch process images')
        batch_parser.add_argument('input_dir', help='Input directory')
        batch_parser.add_argument('--output', '-o', required=True, help='Output directory')
        batch_parser.add_argument('--operation', choices=[
            'enhance', 'denoise', 'sharpen', 'blur', 'edge_detect'
        ], default='enhance', help='Processing operation')
        batch_parser.add_argument('--recursive', '-r', action='store_true',
                                help='Process subdirectories recursively')
        batch_parser.add_argument('--workers', type=int, help='Number of worker threads')
        batch_parser.add_argument('--format', choices=['jpg', 'png', 'tiff'], 
                                help='Output format')
        batch_parser.add_argument('--quality', type=int, help='Output quality (1-100)')
    
    def _add_ml_enhance_parser(self, subparsers) -> None:
        """Add ML enhancement command parser."""
        ml_parser = subparsers.add_parser('ml-enhance', 
                                        help='ML-based image enhancement')
        ml_parser.add_argument('input', help='Input image path')
        ml_parser.add_argument('--output', '-o', help='Output image path')
        ml_parser.add_argument('--model', choices=['esrgan', 'dncnn', 'enlightengan'],
                             default='esrgan', help='ML model to use')
        ml_parser.add_argument('--model-path', help='Path to custom model weights')
        ml_parser.add_argument('--scale', type=int, default=2, 
                             help='Super-resolution scale factor')
        ml_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'],
                             default='auto', help='Processing device')
    
    def _add_style_transfer_parser(self, subparsers) -> None:
        """Add style transfer command parser."""
        style_parser = subparsers.add_parser('style-transfer', 
                                           help='Neural style transfer')
        style_parser.add_argument('input', help='Content image path')
        style_parser.add_argument('--output', '-o', help='Output image path')
        style_parser.add_argument('--style-strength', type=float, default=1.0,
                                help='Style transfer strength (0.0-1.0)')
        style_parser.add_argument('--model-path', help='Path to style model')
    
    def _add_analyze_parser(self, subparsers) -> None:
        """Add analysis command parser."""
        analyze_parser = subparsers.add_parser('analyze', 
                                             help='Analyze image quality')
        analyze_parser.add_argument('original', help='Original image path')
        analyze_parser.add_argument('processed', nargs='?', 
                                  help='Processed image path (optional)')
        analyze_parser.add_argument('--report', help='Save report to file')
        analyze_parser.add_argument('--format', choices=['text', 'json'], 
                                  default='text', help='Report format')
    
    def _add_config_parser(self, subparsers) -> None:
        """Add configuration command parser."""
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action')
        
        # Show config
        config_subparsers.add_parser('show', help='Show current configuration')
        
        # Set config
        set_parser = config_subparsers.add_parser('set', help='Set configuration value')
        set_parser.add_argument('key', help='Configuration key')
        set_parser.add_argument('value', help='Configuration value')
        
        # Reset config
        config_subparsers.add_parser('reset', help='Reset to default configuration')
        
        # Create presets
        config_subparsers.add_parser('create-presets', help='Create preset configurations')
    
    def _add_model_parser(self, subparsers) -> None:
        """Add model management command parser."""
        model_parser = subparsers.add_parser('models', help='Model management')
        model_subparsers = model_parser.add_subparsers(dest='model_action')
        
        # List models
        model_subparsers.add_parser('list', help='List available models')
        
        # Load model
        load_parser = model_subparsers.add_parser('load', help='Load model')
        load_parser.add_argument('name', help='Model name')
        load_parser.add_argument('--path', help='Model weights path')
        
        # Model info
        info_parser = model_subparsers.add_parser('info', help='Model information')
        info_parser.add_argument('name', help='Model name')
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI application."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Configure logging based on verbosity
        if parsed_args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif parsed_args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load custom config if provided
        if parsed_args.config:
            self.config_manager.load_processing_config(parsed_args.config)
        
        # Enable GPU if requested
        if parsed_args.gpu:
            self.config_manager.processing_config.enable_gpu = True
        
        # Route to appropriate command handler
        try:
            if parsed_args.command == 'enhance':
                return self._handle_enhance(parsed_args)
            elif parsed_args.command == 'batch':
                return self._handle_batch(parsed_args)
            elif parsed_args.command == 'ml-enhance':
                return self._handle_ml_enhance(parsed_args)
            elif parsed_args.command == 'style-transfer':
                return self._handle_style_transfer(parsed_args)
            elif parsed_args.command == 'analyze':
                return self._handle_analyze(parsed_args)
            elif parsed_args.command == 'config':
                return self._handle_config(parsed_args)
            elif parsed_args.command == 'models':
                return self._handle_models(parsed_args)
            else:
                parser.print_help()
                return 1
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if parsed_args.verbose:
                raise
            return 1
    
    def _handle_enhance(self, args) -> int:
        """Handle image enhancement command."""
        import cv2
        
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            suffix = self.config_manager.processing_config.output_suffix
            output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
        
        logger.info(f"Processing: {input_path} -> {output_path}")
        
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            logger.error(f"Failed to load image: {input_path}")
            return 1
        
        # Start timing
        self.performance_monitor.start_timing()
        
        # Apply enhancement based on method
        if args.method == 'auto':
            processed = self.adaptive_enhancer.auto_enhance(image)
        elif args.method == 'low_light':
            algorithm = args.algorithm or 'retinex'
            processed = self.advanced_enhancer.enhance_low_light(image, algorithm)
        elif args.method == 'contrast':
            algorithm = args.algorithm or 'adaptive'
            processed = self.advanced_enhancer.enhance_contrast(image, algorithm)
        elif args.method == 'color':
            algorithm = args.algorithm or 'vibrance'
            processed = self.advanced_enhancer.enhance_color(image, algorithm)
        else:
            # Use core processor
            processed = self.image_processor.process(image, args.method)
        
        # Record performance
        duration = self.performance_monitor.stop_timing()
        self.performance_monitor.log_operation('enhance', duration, image.shape)
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), processed)
        
        if success:
            logger.info(f"Enhanced image saved: {output_path}")
            logger.info(f"Processing time: {duration:.3f}s")
            
            # Calculate and display metrics if verbose
            if args.verbose:
                metrics = self.metrics_calculator.calculate_image_metrics(image, processed)
                report = self.metrics_calculator.generate_quality_report(metrics)
                print("\n" + report)
            
            return 0
        else:
            logger.error(f"Failed to save image: {output_path}")
            return 1
    
    def _handle_batch(self, args) -> int:
        """Handle batch processing command."""
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output)
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 1
        
        logger.info(f"Batch processing: {input_dir} -> {output_dir}")
        
        # Start timing
        self.performance_monitor.start_timing()
        
        # Process directory
        results = self.batch_processor.process_directory(
            input_dir, output_dir, args.operation
        )
        
        # Record performance
        duration = self.performance_monitor.stop_timing()
        
        # Display results
        logger.info(f"Batch processing completed in {duration:.3f}s")
        logger.info(f"Processed: {results['processed']} images")
        logger.info(f"Failed: {results['failed']} images")
        
        if results['failed'] > 0:
            logger.warning(f"Some images failed to process")
        
        return 0 if results['failed'] == 0 else 1
    
    def _handle_ml_enhance(self, args) -> int:
        """Handle ML-based enhancement command."""
        try:
            import cv2
            
            # Load and prepare model
            if args.model not in self.model_manager.list_models():
                logger.error(f"Model {args.model} not available")
                return 1
            
            # Load model if not already loaded
            try:
                self.model_manager.load_model(args.model, args.model_path)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return 1
            
            # Load and process image
            input_path = Path(args.input)
            image = cv2.imread(str(input_path))
            if image is None:
                logger.error(f"Failed to load image: {input_path}")
                return 1
            
            logger.info(f"Processing with {args.model} model...")
            
            # Start timing
            self.performance_monitor.start_timing()
            
            # Process image
            processed = self.model_manager.predict(args.model, image)
            
            # Record performance
            duration = self.performance_monitor.stop_timing()
            
            # Save result
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = input_path.parent / f"{input_path.stem}_ml_enhanced{input_path.suffix}"
            
            success = cv2.imwrite(str(output_path), processed)
            
            if success:
                logger.info(f"ML-enhanced image saved: {output_path}")
                logger.info(f"Processing time: {duration:.3f}s")
                return 0
            else:
                logger.error(f"Failed to save image: {output_path}")
                return 1
                
        except ImportError:
            logger.error("ML features not available. Install PyTorch to use ML models.")
            return 1
    
    def _handle_style_transfer(self, args) -> int:
        """Handle style transfer command."""
        try:
            import cv2
            
            # Load style transfer model
            if 'style_transfer' not in self.model_manager.list_models():
                logger.error("Style transfer model not available")
                return 1
            
            self.model_manager.load_model('style_transfer', args.model_path)
            
            # Load and process image
            input_path = Path(args.input)
            image = cv2.imread(str(input_path))
            if image is None:
                logger.error(f"Failed to load image: {input_path}")
                return 1
            
            logger.info("Applying style transfer...")
            
            # Start timing
            self.performance_monitor.start_timing()
            
            # Apply style transfer
            style_transfer = self.model_manager.models['style_transfer']
            processed = style_transfer.transfer_style(image, args.style_strength)
            
            # Record performance
            duration = self.performance_monitor.stop_timing()
            
            # Save result
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = input_path.parent / f"{input_path.stem}_stylized{input_path.suffix}"
            
            success = cv2.imwrite(str(output_path), processed)
            
            if success:
                logger.info(f"Stylized image saved: {output_path}")
                logger.info(f"Processing time: {duration:.3f}s")
                return 0
            else:
                logger.error(f"Failed to save image: {output_path}")
                return 1
                
        except ImportError:
            logger.error("Style transfer not available. Install PyTorch.")
            return 1
    
    def _handle_analyze(self, args) -> int:
        """Handle image analysis command."""
        import cv2
        
        original_path = Path(args.original)
        if not original_path.exists():
            logger.error(f"Original image not found: {original_path}")
            return 1
        
        original = cv2.imread(str(original_path))
        if original is None:
            logger.error(f"Failed to load original image: {original_path}")
            return 1
        
        if args.processed:
            processed_path = Path(args.processed)
            if not processed_path.exists():
                logger.error(f"Processed image not found: {processed_path}")
                return 1
            
            processed = cv2.imread(str(processed_path))
            if processed is None:
                logger.error(f"Failed to load processed image: {processed_path}")
                return 1
            
            # Calculate comparison metrics
            metrics = self.metrics_calculator.calculate_image_metrics(original, processed)
            
            if args.format == 'json':
                report = json.dumps(metrics, indent=2)
            else:
                report = self.metrics_calculator.generate_quality_report(metrics)
        else:
            # Analyze single image
            if args.format == 'json':
                analysis = self.adaptive_enhancer._analyze_image(original)
                report = json.dumps(analysis, indent=2)
            else:
                analysis = self.adaptive_enhancer._analyze_image(original)
                report = f"Image Analysis Report\n"
                report += f"====================\n\n"
                for key, value in analysis.items():
                    report += f"{key.replace('_', ' ').title()}: {value}\n"
        
        # Output report
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            logger.info(f"Report saved: {args.report}")
        else:
            print(report)
        
        return 0
    
    def _handle_config(self, args) -> int:
        """Handle configuration commands."""
        if args.config_action == 'show':
            config_dict = {
                'processing': self.config_manager.processing_config.__dict__,
                'model': self.config_manager.model_config.__dict__
            }
            print(json.dumps(config_dict, indent=2))
        
        elif args.config_action == 'set':
            # Simple key-value setting (could be enhanced)
            if hasattr(self.config_manager.processing_config, args.key):
                setattr(self.config_manager.processing_config, args.key, args.value)
                self.config_manager.save_processing_config()
                logger.info(f"Set {args.key} = {args.value}")
            else:
                logger.error(f"Unknown configuration key: {args.key}")
                return 1
        
        elif args.config_action == 'reset':
            from src.utils import ProcessingConfig
            self.config_manager.processing_config = ProcessingConfig()
            self.config_manager.save_processing_config()
            logger.info("Configuration reset to defaults")
        
        elif args.config_action == 'create-presets':
            self.config_manager.create_preset_configs()
            logger.info("Preset configurations created")
        
        return 0
    
    def _handle_models(self, args) -> int:
        """Handle model management commands."""
        if args.model_action == 'list':
            models = self.model_manager.list_models()
            print("Available models:")
            for model in models:
                info = self.model_manager.get_model_info(model)
                status = "✓ Loaded" if info['is_loaded'] else "○ Not loaded"
                print(f"  {model} ({info['type']}) - {status}")
        
        elif args.model_action == 'load':
            try:
                self.model_manager.load_model(args.name, args.path)
                logger.info(f"Successfully loaded model: {args.name}")
            except Exception as e:
                logger.error(f"Failed to load model {args.name}: {e}")
                return 1
        
        elif args.model_action == 'info':
            try:
                info = self.model_manager.get_model_info(args.name)
                print(json.dumps(info, indent=2))
            except Exception as e:
                logger.error(f"Model {args.name} not found: {e}")
                return 1
        
        return 0


def main() -> int:
    """Main entry point."""
    cli = ImageProcessingCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
