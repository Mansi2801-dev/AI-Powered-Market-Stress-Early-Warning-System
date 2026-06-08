#### AI-Powered Market Stress Early Warning System

### Overview
This project predicts future market stress using order book dynamics and a Transformer-based deep learning model.

The system learns patterns from historical order book features and predicts a Future Stress Score at a future horizon of 30 steps ahead.

The predicted stress score is then mapped into market regimes:
1. NORMAL
2. WARNING
3. HIGH STRESS
4. CRITICAL

The project includes:
1. Feature engineering pipeline
2. Future stress score generation
3. Sequence dataset creation
4. Time-series train/validation/test split
5. Transformer model training
6. Streamlit dashboard for inference

## Features Used

The model uses four order book features:
1. Spread
2. L10_log (Liquidity of 10 levels)
3. L50_log (Liquidity of 50 levels)
4. Imbalance

## Input shape:
60 time steps × 4 features

## Model Architecture

# Transformer-based regression model.

-  Configuration:
1. d_model = 64
2. nhead = 4
3. n_layers = 2
4. dim_feedforward = 2048
5. lookback window = 60
6. loss function = HuberLoss
7. optimizer = AdamW

## Output:
Single Future Stress Score

## Clone the repository:

git clone <repository-url>
cd AI-Powered-Market-Stress-Early-Warning-System

## Create a virtual environment:
    python -m venv venv

## Activate environment
- Windows: venv\Scripts\activate
- Linux/Mac: source venv/bin/activate

## Install dependencies
pip install -r Requirements.txt

## Project Structure
AI-Powered-Market-Stress-Early-Warning-System/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── stress_model.pth
│
├── src/
│   ├── preprocess.py
│   ├── labels.py
│   ├── sequence_dataset.py
│   ├── split_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── stress_threshold.py
│   └── app.py
|   └── features.py  
|
├── requirements.txt
└── README.md

## Dataset
Make a folder named 'Data', inside it make 2 subfolders 'raw' and 'processed'. 

The raw order book dataset is not included in this repository due to size constraints.
Download it from:
<https://www.kaggle.com/datasets/ilyazawilsiv/cryptocurrency-order-book-data-asks-and-bids>

Place the downloaded file(s) inside:
data/raw/

All other processed dataset should be stored in subfolder 'processed'.

### Complete Pipeline Execution Order

Run the project in the following order.

## Step 1 - Preprocess the raw dataset

Input :
The raw dataset 

Run :
pyhton preprocess.py

Output:
The processed dataset is saved as clean_dataset.parquet

## Step 2 - Generating Features

Input :
data/processed/clean_dataset.parquet

Run :
python features.py

Output :
The features to be used for training transformer in future are generated and stored in features.parquet

## Step 3 — Generate Future Stress Labels

Input:
data/processed/features.parquet

Run:
python labels.py

Output:
data/processed/training_dataset.parquet

Purpose:
Creates future market stress labels using:
1. future spread
2. future liquidity
3. future imbalance

and constructs:
-> Future_Stress_Score

## Step 4 — Build Sequence Dataset

Input:
data/processed/training_dataset.parquet

Run:
python sequence_dataset.py

Output:
data/processed/X.npy
data/processed/Y.npy

Purpose:
Converts tabular features into rolling sequences.

Input shape:
(60, 4)

## Step 5 — Time-Series Train/Validation/Test Split

Input:
data/processed/X.npy
data/processed/Y.npy

Run:
python split_dataset.py

Output:
X_train.npy
Y_train.npy

X_val.npy
Y_val.npy

X_test.npy
Y_test.npy

Split:

70% Train
15% Validation
15% Test

### Important:
The split is chronological to avoid data leakage.

## Step 6 - Creating Stress threshholds

Input:
data/processed/training_dataset.parquet

Run : 
python stress_threshold.py
It creates quantile scores that is used to predict the level of stress

## Step 7 - MarketStressTransformer Architecture

Run :
python model.py

## Step 8 — Train Transformer Model

Run:
python train.py

Output:
models/stress_model.pth

Training includes:
1. Huber Loss
2. AdamW optimizer
3. Gradient clipping
4. Validation monitoring
5. Learning rate scheduling

## Step 9 — Launch Dashboard

Run:
streamlit run src/app.py

The dashboard will:
1. Load trained model
2. Generate predictions
3. Display stress score
4. Display market regime
5. Market Regimes

Current thresholds:
Regime	Score Range
NORMAL	< -2.0645
WARNING	-2.0645 to -1.2331
HIGH STRESS	-1.2331 to 0.3668
CRITICAL	> 0.3668

These thresholds were derived from the training data distribution.

### Notes

1. Current dashboard uses a placeholder input generator:
    np.random.randn(60,4)

2. To deploy in production:
    Replace the placeholder with a live order book feed.

3. The model will then generate real-time stress forecasts.