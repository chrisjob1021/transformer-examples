from setuptools import setup, find_packages

setup(
    name="transformer-examples",
    version="0.1.0",
    description="Transformer architecture implementations and examples",
    author="Chris O'Brien",
    author_email="your.email@example.com",  # Replace with your email
    packages=find_packages(),
    install_requires=[
        "torch>=1.0.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "debugpy>=1.0.0",
    ],
    python_requires=">=3.6",
)