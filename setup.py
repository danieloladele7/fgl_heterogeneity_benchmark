from setuptools import setup, find_packages

setup(
    name="fgl_heterogeneity",
    version="0.1.0",
    description="Benchmarking suite for non-IID data in Federated Graph Learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=2.6",
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "python-louvain>=0.16",   # for community detection
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
