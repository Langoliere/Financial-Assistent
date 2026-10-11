# Training Scripts

This directory contains scripts for data preparation, model training, and evaluation.

## Prerequisites

Ensure you have all dependencies installed:
```bash
pip install -e .
```

## Training Pipeline

### Step 1: Prepare Data

Download historical OHLCV data from MOEX:

```bash
python training/prepare_data.py
```

This will:
- Fetch 10 years of daily SBER data from Moscow Exchange
- Save to `data/sber_ohlcv.csv`
- Print data summary

### Step 2: Train Model

Train the LSTM direction prediction model:

```bash
python training/train_direction.py
```

This will:
- Load data from `data/sber_ohlcv.csv`
- Split chronologically: 80% train, 10% validation, 10% test
- Engineer features using FeaturePipeline
- Train LSTM model with early stopping
- Calculate optimal classification threshold
- Evaluate on test set
- Save artifacts to `artifacts/direction_model/`

### Step 3: Evaluate Model

Evaluate the trained model:

```bash
python training/evaluate.py
```

This will:
- Load trained model and artifacts
- Load test data
- Evaluate on test set
- Print metrics and classification report
- Compare with saved metadata

## Artifacts

After training, the following artifacts are saved to `artifacts/direction_model/`:

- `model.keras` - Trained Keras model
- `scaler_X.pkl` - Fitted MinMaxScaler for features
- `feature_schema.json` - Feature order and types
- `threshold.json` - Optimal classification threshold
- `metadata.json` - Training metadata (dates, metrics, hyperparameters)

## Notes

- All scripts use fixed random seeds for reproducibility
- Data splits are chronological (no shuffling)
- No data leakage: features use only current and past observations
- Pipeline is fitted only on train data
- Validation and test use context from previous segments for sequence generation
