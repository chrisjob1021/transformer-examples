from setuptools import setup, find_packages

setup(
    name="transformer-examples",
    version="0.1.0",
    description="Transformer architecture toy implementations and examples",
    author="Chris O'Brien",
    author_email="chris@chrisobrien.ai",
    packages=find_packages(include=['models', 'models.*', 'attention', 'attention.*', 'utils', 'utils.*']),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
)