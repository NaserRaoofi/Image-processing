"""
FastAPI Web Application for Image Processing
===========================================

Production-ready REST API with comprehensive image processing capabilities.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import io
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import logging
import uuid
from datetime import datetime
from pydantic import BaseModel

# Import our modules
from src.core import ImageProcessor, BatchProcessor, CoreProcessingConfig
from src.enhancement import AdvancedEnhancer, AdaptiveEnhancer
from src.utils import ConfigManager, MetricsCalculator, PerformanceMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Advanced Image Processing API",
    description="Enterprise-grade image processing with ML/DL capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processing components
config_manager = ConfigManager()
image_processor = ImageProcessor(config_manager.processing_config)
batch_processor = BatchProcessor(image_processor)
advanced_enhancer = AdvancedEnhancer()
adaptive_enhancer = AdaptiveEnhancer()
metrics_calculator = MetricsCalculator()
performance_monitor = PerformanceMonitor()

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API
class ProcessingRequest(BaseModel):
    operation: str = "enhance"
    method: Optional[str] = "auto"
    algorithm: Optional[str] = None
    strength: float = 1.0
    preserve_colors: bool = True


class ProcessingResponse(BaseModel):
    success: bool
    message: str
    processing_time: float
    output_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class BatchRequest(BaseModel):
    operation: str = "enhance"
    format: str = "jpg"
    quality: int = 95
    recursive: bool = False


class AnalysisResponse(BaseModel):
    success: bool
    metrics: Dict[str, float]
    quality_assessment: str
    recommendations: List[str]


# Job tracking for async operations
active_jobs: Dict[str, Dict[str, Any]] = {}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Advanced Image Processing API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Single image enhancement",
            "Batch processing",
            "ML-based enhancement",
            "Quality analysis",
            "Style transfer",
            "Real-time processing"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(active_jobs)
    }


@app.post("/enhance", response_model=ProcessingResponse)
async def enhance_image(
    file: UploadFile = File(...),
    operation: str = Form("enhance"),
    method: str = Form("auto"),
    algorithm: Optional[str] = Form(None),
    strength: float = Form(1.0),
    preserve_colors: bool = Form(True),
    return_base64: bool = Form(False)
):
    """
    Enhance a single image with various algorithms.
    
    Parameters:
    - file: Image file to process
    - operation: Type of operation (enhance, denoise, sharpen, etc.)
    - method: Enhancement method (auto, clahe, adaptive, etc.)
    - algorithm: Specific algorithm for the method
    - strength: Enhancement strength (0.1-2.0)
    - preserve_colors: Whether to preserve original colors
    - return_base64: Return image as base64 string instead of file
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Start timing
        performance_monitor.start_timing()
        
        # Apply processing based on method
        if method == 'auto':
            processed = adaptive_enhancer.auto_enhance(image)
        elif method == 'low_light':
            algorithm = algorithm or 'retinex'
            processed = advanced_enhancer.enhance_low_light(image, algorithm)
        elif method == 'contrast':
            algorithm = algorithm or 'adaptive'
            processed = advanced_enhancer.enhance_contrast(image, algorithm)
        elif method == 'color':
            algorithm = algorithm or 'vibrance'
            processed = advanced_enhancer.enhance_color(image, algorithm)
        else:
            processed = image_processor.process(image, operation)
        
        # Record performance
        processing_time = performance_monitor.stop_timing()
        performance_monitor.log_operation(operation, processing_time, image.shape)
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_image_metrics(image, processed)
        
        if return_base64:
            # Return as base64 string
            _, buffer = cv2.imencode('.jpg', processed)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return ProcessingResponse(
                success=True,
                message="Image processed successfully",
                processing_time=processing_time,
                output_path=f"data:image/jpeg;base64,{img_base64}",
                metrics=metrics
            )
        else:
            # Save to temporary file
            output_filename = f"{uuid.uuid4()}.jpg"
            output_path = UPLOAD_DIR / output_filename
            
            cv2.imwrite(str(output_path), processed)
            
            return ProcessingResponse(
                success=True,
                message="Image processed successfully",
                processing_time=processing_time,
                output_path=f"/download/{output_filename}",
                metrics=metrics
            )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance-async")
async def enhance_image_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    operation: str = Form("enhance"),
    method: str = Form("auto")
):
    """
    Enhance image asynchronously and return job ID.
    """
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    input_filename = f"{job_id}_input{Path(file.filename).suffix}"
    input_path = UPLOAD_DIR / input_filename
    
    with open(input_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Initialize job tracking
    active_jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "operation": operation,
        "method": method,
        "input_file": str(input_path),
        "progress": 0
    }
    
    # Add background task
    background_tasks.add_task(
        process_image_background,
        job_id,
        str(input_path),
        operation,
        method
    )
    
    return {"job_id": job_id, "status": "queued"}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an async job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]


@app.post("/batch")
async def batch_process(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    operation: str = Form("enhance"),
    format: str = Form("jpg"),
    quality: int = Form(95)
):
    """
    Process multiple images in batch.
    """
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    # Save all uploaded files
    input_files = []
    for i, file in enumerate(files):
        if not file.content_type.startswith('image/'):
            continue
        
        filename = f"input_{i}{Path(file.filename).suffix}"
        file_path = job_dir / filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        input_files.append(str(file_path))
    
    # Initialize job tracking
    active_jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "operation": operation,
        "total_files": len(input_files),
        "processed_files": 0,
        "failed_files": 0,
        "progress": 0
    }
    
    # Add background task
    background_tasks.add_task(
        process_batch_background,
        job_id,
        input_files,
        operation,
        format,
        quality
    )
    
    return {"job_id": job_id, "status": "queued", "total_files": len(input_files)}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None)
):
    """
    Analyze image quality and provide recommendations.
    """
    try:
        # Read main image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        if reference_file:
            # Compare with reference image
            ref_data = await reference_file.read()
            ref_nparr = np.frombuffer(ref_data, np.uint8)
            reference = cv2.imdecode(ref_nparr, cv2.IMREAD_COLOR)
            
            if reference is None:
                raise HTTPException(status_code=400, detail="Invalid reference image format")
            
            metrics = metrics_calculator.calculate_image_metrics(reference, image)
            
            # Generate recommendations based on metrics
            recommendations = []
            if metrics.get('ssim', 0) < 0.8:
                recommendations.append("Consider noise reduction or sharpening")
            if metrics.get('contrast_ratio', 1) < 0.8:
                recommendations.append("Image may benefit from contrast enhancement")
            if metrics.get('brightness_change', 0) < -20:
                recommendations.append("Image appears underexposed, try brightness enhancement")
            
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
        
        else:
            # Analyze single image
            analysis = adaptive_enhancer._analyze_image(image)
            metrics = analysis
            
            # Generate recommendations
            recommendations = []
            if analysis.get('is_low_light', False):
                recommendations.append("Try low-light enhancement with Retinex algorithm")
            if analysis.get('contrast', 0) < 30:
                recommendations.append("Image would benefit from contrast enhancement")
            if analysis.get('is_low_saturation', False):
                recommendations.append("Consider color enhancement to improve vibrancy")
            
            # Quality assessment based on analysis
            if analysis.get('mean_brightness', 0) > 200:
                quality = "Overexposed"
            elif analysis.get('mean_brightness', 0) < 50:
                quality = "Underexposed"
            elif analysis.get('contrast', 0) > 60:
                quality = "High contrast"
            elif analysis.get('contrast', 0) < 20:
                quality = "Low contrast"
            else:
                quality = "Balanced"
        
        return AnalysisResponse(
            success=True,
            metrics=metrics,
            quality_assessment=quality,
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed image."""
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/config")
async def get_config():
    """Get current processing configuration."""
    return {
        "processing": config_manager.processing_config.__dict__,
        "model": config_manager.model_config.__dict__
    }


@app.post("/config")
async def update_config(config_data: Dict[str, Any]):
    """Update processing configuration."""
    try:
        for key, value in config_data.items():
            if hasattr(config_manager.processing_config, key):
                setattr(config_manager.processing_config, key, value)
        
        config_manager.save_processing_config()
        return {"success": True, "message": "Configuration updated"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/performance")
async def get_performance_stats():
    """Get performance statistics."""
    stats = image_processor.get_stats()
    return {
        "processing_stats": stats,
        "active_jobs": len(active_jobs),
        "server_status": "operational"
    }


# Background task functions
async def process_image_background(job_id: str, input_path: str, operation: str, method: str):
    """Background task for async image processing."""
    try:
        active_jobs[job_id]["status"] = "processing"
        active_jobs[job_id]["progress"] = 25
        
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise Exception("Failed to load image")
        
        active_jobs[job_id]["progress"] = 50
        
        # Process image
        if method == 'auto':
            processed = adaptive_enhancer.auto_enhance(image)
        else:
            processed = image_processor.process(image, operation)
        
        active_jobs[job_id]["progress"] = 75
        
        # Save result
        output_filename = f"{job_id}_output.jpg"
        output_path = UPLOAD_DIR / output_filename
        cv2.imwrite(str(output_path), processed)
        
        active_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "output_file": f"/download/{output_filename}",
            "completed_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


async def process_batch_background(job_id: str, input_files: List[str], 
                                 operation: str, format: str, quality: int):
    """Background task for batch processing."""
    try:
        active_jobs[job_id]["status"] = "processing"
        
        job_dir = UPLOAD_DIR / job_id
        output_dir = job_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        for i, input_file in enumerate(input_files):
            try:
                # Load and process image
                image = cv2.imread(input_file)
                if image is None:
                    failed_count += 1
                    continue
                
                processed = image_processor.process(image, operation)
                
                # Save result
                filename = f"output_{i}.{format}"
                output_path = output_dir / filename
                cv2.imwrite(str(output_path), processed)
                
                processed_count += 1
                
                # Update progress
                progress = int((i + 1) / len(input_files) * 100)
                active_jobs[job_id].update({
                    "progress": progress,
                    "processed_files": processed_count,
                    "failed_files": failed_count
                })
            
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                failed_count += 1
        
        active_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now().isoformat(),
            "output_directory": f"/download/{job_id}/output/"
        })
    
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
