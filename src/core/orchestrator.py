name: AEGIS Alpha Signal Pipeline

on:
  schedule:
    - cron: '0 * * * *'  # Run every hour
  workflow_dispatch:      # Allow manual runs

jobs:
  fetch-data:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Dependencies
        run: |
          pip install --upgrade pip
          pip install pandas ccxt pyyaml pyarrow tenacity

      - name: Prepare Package Structure
        run: |
          touch __init__.py
          mkdir -p src/core src/indicators src/utils src/notifications
          touch src/__init__.py src/core/__init__.py src/indicators/__init__.py
          touch src/utils/__init__.py src/notifications/__init__.py

      - name: Run Data Fetcher
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)"
          python src/core/data_fetcher.py

      - name: Upload Raw Data
        uses: actions/upload-artifact@v4
        with:
          name: raw-ohlcv
          path: data/raw/
          retention-days: 1

  feature-engineering:
    needs: fetch-data
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Dependencies
        run: |
          pip install --upgrade pip
          pip install pandas numpy pyyaml pyarrow pandas-ta tenacity ccxt

      - name: Prepare Package Structure
        run: |
          touch __init__.py
          touch src/__init__.py src/core/__init__.py src/indicators/__init__.py

      - name: Download Raw Data
        uses: actions/download-artifact@v4
        with:
          name: raw-ohlcv
          path: data/raw/

      - name: Process Features
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)"
          python << 'PYEOF'
          import sys
          import os
          from pathlib import Path
          import pandas as pd
          
          # Force root into path
          sys.path.insert(0, os.getcwd())
          
          from src.core.feature_engineering import FeatureEngineer
          
          engineer = FeatureEngineer()
          raw_dir = Path("data/raw")
          proc_dir = Path("data/processed")
          proc_dir.mkdir(parents=True, exist_ok=True)
          
          for file in raw_dir.glob("*.parquet"):
              print(f"Engineering: {file.name}")
              df = pd.read_parquet(file)
              processed_df = engineer.calculate_all_features(df)
              processed_df.to_parquet(proc_dir / file.name)
          PYEOF

      - name: Upload Processed Data
        uses: actions/upload-artifact@v4
        with:
          name: processed-features
          path: data/processed/
          retention-days: 1

  generate-signals:
    needs: feature-engineering
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Dependencies
        run: |
          pip install --upgrade pip
          pip install pandas pyyaml pyarrow tenacity ccxt

      - name: Download Processed Data
        uses: actions/download-artifact@v4
        with:
          name: processed-features
          path: data/processed/

      - name: Run Signal Engine
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)"
          python src/core/signal_generator.py

      - name: Upload Signals
        uses: actions/upload-artifact@v4
        with:
          name: aegis-signals
          path: signals/
          retention-days: 7
