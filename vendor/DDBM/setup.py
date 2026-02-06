from setuptools import setup, find_packages

setup(
    name="ddbm",
    version="0.4.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy",
        "pillow",
        "tqdm",
        # Diffusers integration (core dependencies)
        'diffusers>=0.25.0',
        'transformers>=4.48.0',  # Security fix for deserialization vulnerability
        'accelerate>=0.20.0',
        # Additional utilities
        "scipy",
        "packaging",
    ],
    extras_require={
        "training": [
            # Dependencies for training scripts
            "wandb",
            'torchmetrics[image]',
        ],
        "evaluation": [
            # Dependencies for evaluation
            'clean-fid==0.1.35',
            'torchmetrics[image]',
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },
    python_requires=">=3.8",
    author="DDBM Authors",
    description="Denoising Diffusion Bridge Models with Hugging Face diffusers integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alexzhou907/DDBM",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="diffusion, bridge models, image-to-image, deep learning, huggingface, diffusers",
)
