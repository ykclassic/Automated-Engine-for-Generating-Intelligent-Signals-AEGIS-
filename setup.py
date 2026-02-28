"""
AEGIS - Automated Engine for Generating Intelligent Signals
Setup script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aegis",
    version="1.0.0",
    author="AEGIS Team",
    description="Institutional-grade cryptocurrency signal generation bot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aegis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "lightgbm>=4.0.0",
            "xgboost>=1.7.0",
            "shap>=0.42.0",
        ],
        "dashboard": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
        ],
    },
)
