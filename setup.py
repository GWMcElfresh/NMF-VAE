from setuptools import setup, find_packages

setup(
    name="nmfvae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "anndata>=0.8.0",
        "scanpy>=1.9.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "umap-learn>=0.5.0",
    ],
    python_requires=">=3.8",
    author="NMF-VAE Authors",
    description="NMF-like Variational Autoencoder for single-cell RNA-seq data",
)
