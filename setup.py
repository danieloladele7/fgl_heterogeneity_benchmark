from setuptools import find_packages, setup

setup(
    name="fgl_heterogeneity",
    version="0.2.0",
    description="Benchmarking suite for graph heterogeneity in federated graph learning",
    author="OpenAI",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=2.8",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "jupyter>=1.0.0", "nbformat>=5.0.0"],
        "torchgeo": ["torch>=1.10.0", "torch-geometric>=2.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
