from setuptools import setup, find_packages

setup(
    name="nmfvae",
    version="0.1.0",
    packages=find_packages(include=["model", "model.*", "utils", "utils.*"]),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "anndata>=0.9.0",
        "scanpy>=1.9.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "umap-learn>=0.5.0",
        "requests>=2.28.0",
        "mygene>=3.2.2",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "nmfvae-train=scripts.train:main",
            "nmfvae-preprocess=scripts.preprocess:main",
        ],
    },
    author="NMF-VAE Authors",
    description="NMF-like Variational Autoencoder for single-cell RNA-seq data",
)
