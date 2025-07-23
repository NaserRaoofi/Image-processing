"""
Setup configuration for the Advanced Image Processing package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-image-processing",
    version="2.0.0",
    author="Advanced Computer Vision Team",
    author_email="team@imageprocessing.ai",
    description="Enterprise-grade image processing library with state-of-the-art ML/DL capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/NaserRaoofi/Image-processing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "ml": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "ultralytics>=8.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest-cov>=4.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "python-multipart>=0.0.6",
        ]
    },
    entry_points={
        "console_scripts": [
            "image-process=cli:main",
            "img-enhance=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    keywords=[
        "image processing",
        "computer vision", 
        "machine learning",
        "deep learning",
        "opencv",
        "enhancement",
        "super resolution",
        "denoising",
        "enterprise",
        "production"
    ],
    project_urls={
        "Bug Reports": "https://github.com/NaserRaoofi/Image-processing/issues",
        "Source": "https://github.com/NaserRaoofi/Image-processing",
        "Documentation": "https://docs.imageprocessing.ai",
        "Funding": "https://github.com/sponsors/NaserRaoofi",
    },
)
