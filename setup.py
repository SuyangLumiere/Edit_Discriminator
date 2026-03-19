from setuptools import setup, find_packages
from setuptools import setup, find_packages

setup(
    name="Qwen3VLAuditor",
    version="0.1.0",
    author="Rico", 
    description="A Qwen3-VL based image edit fidelity scorer with early exit and dynamic routing.",
    packages=find_packages(),           
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
    ],
    python_requires=">=3.10",              
)
