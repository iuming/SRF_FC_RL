"""
Setup script for RF Cavity Control System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() 
                   if line.strip() and not line.startswith('#')]
    return []

setup(
    name="rf-cavity-control",
    version="1.0.0",
    author="Ming Liu",
    author_email="ming.liu@example.com",
    description="A reinforcement learning system for RF cavity control using PPO",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/iuming/ML_Learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rf-cavity-train=scripts.train_rf_cavity:main",
            "rf-cavity-test=scripts.test_rf_cavity:main",
            "rf-cavity-realtime=scripts.realtime_simple:main",
            "rf-cavity-gui=scripts.realtime_gui:main",
        ],
    },
    include_package_data=True,
    package_data={
        "configs": ["*.py"],
        "": ["*.md", "*.txt", "*.bat"],
    },
    keywords="reinforcement-learning rf-cavity control ppo physics simulation",
    project_urls={
        "Bug Reports": "https://github.com/iuming/ML_Learning/issues",
        "Source": "https://github.com/iuming/ML_Learning",
        "Documentation": "https://github.com/iuming/ML_Learning/blob/main/RL_Learning/custom/20250725/README.md",
    },
)
