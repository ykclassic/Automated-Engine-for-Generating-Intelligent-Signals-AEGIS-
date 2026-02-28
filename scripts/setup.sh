#!/bin/bash

# AEGIS Setup Script
# Run this to initialize the development environment

set -e

echo "🛡️  Setting up AEGIS Development Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{raw,processed,cache,models}
mkdir -p logs
mkdir -p notebooks
mkdir -p tests

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# AEGIS Environment Configuration
# Copy this to .env and fill in your values

# Exchange API Keys (optional for public data)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here

# Discord Webhook for notifications
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF
fi

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.env
*.egg-info/

# Data
data/raw/*.parquet
data/processed/*.parquet
data/cache/
data/models/*.pkl
data/models/*.joblib

# Logs
logs/*.log

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
fi

# Run initial data fetch test
echo "Testing data pipeline..."
python -c "
from src.core.data_fetcher import fetch_data
import logging
logging.basicConfig(level=logging.INFO)
try:
    df = fetch_data('BTC/USDT', '1h')
    print(f'✅ Successfully fetched {len(df)} candles')
except Exception as e:
    print(f'⚠️  Test fetch failed: {e}')
    print('This is normal if you have no internet connection.')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Test: python -m pytest tests/"
echo "4. Fetch data: python src/core/data_fetcher.py"
