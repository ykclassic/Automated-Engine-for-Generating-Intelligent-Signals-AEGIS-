# Contributing to AEGIS

First off, thank you for considering contributing to AEGIS! It's people like you that make this project better for everyone.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our commitment to:

- **Respect**: Treat everyone with respect. Healthy debate is encouraged, but harassment is not tolerated.
- **Constructive Feedback**: Provide and receive feedback gracefully.
- **Focus**: Keep discussions focused on improving the project.
- **Inclusivity**: Welcome newcomers and help them learn.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- GitHub account
- Code editor (VS Code recommended)

### Setup Development Environment

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/aegis.git
cd aegis

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Install pre-commit hooks
pre-commit install

# 6. Verify setup
pytest tests/ -v
