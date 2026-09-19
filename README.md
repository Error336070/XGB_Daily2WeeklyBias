# XGBoost Daily to Weekly Bias Forecasting Model

This project is a simple example of an XGBoost model implementation to forecast weekly directional bias in financial markets using daily aggregated features. 

## Project Overview

The model uses a set of technical indicators derived from daily price action and volatility to predict either:
1.  **Weekly Directional Bias (Classification)**: Predicts if the upcoming week will be bullish (1) or bearish (0).
2.  **Weekly Return (Regression)**: Predicts the magnitude of the return for the upcoming week.

The example features intentionally don't have predictive value, feel free to replace with custom features for actual financial forecasts.

The primary input data is expected in a CSV file named `FeatureData.csv`.

## Features

The model utilizes various features, excluding `DateTime`. The last two columns in `FeatureData.csv`, `Y_Return5` and `Y_Direction`, are treated as target variables.

-   **Input Features**: Return1, Return5, SmaNorm, RSI14, DeltaRSI, ATR14, StdDev20, ZClose, VolZ, BodyToRange, IntraBarPos, UpperWick, LowerWick.
-   **Target Variables**:
    -   `Y_Direction`: Binary target for classification (e.g., 0 for down, 1 for up).
    -   `Y_Return5`: Continuous target for regression (e.g., 5-period forward return).

## Setup

1.  **Clone the repository (if applicable) or ensure all files are in the same directory.**

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    -   Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your data:**
    -   Ensure you have a `FeatureData.csv` file in the same directory as `xgboost_model.py`.
    -   The CSV file should have the headers as described, including the feature columns and the target columns (`Y_Direction`, `Y_Return5`).
    -   The `DateTime` column will be excluded from features.

## Running the Model

To run the XGBoost model, execute the main script:

```bash
python xgboost_model.py
```

The script will perform the following steps:
1.  Load and preprocess data from `FeatureData.csv`.
2.  Train and evaluate a classification model using `Y_Direction` as the target.
3.  Train and evaluate a regression model using `Y_Return5` as the target.
4.  Both models use walk-forward validation to simulate real-world performance.
5.  Feature importance plots (`feature_importances_classification.png` and `feature_importances_regression.png`) will be saved in the project directory.

## Model Details

-   **Algorithm**: XGBoost (Extreme Gradient Boosting)
-   **Validation**: Walk-forward validation (TimeSeriesSplit-like approach) to respect the temporal order of financial data and simulate out-of-sample performance.
-   **Regularization**: The XGBoost parameters include L1 (alpha) and L2 (lambda) regularization. Early stopping is also used during training to prevent overfitting.
-   **Data Handling**: Missing values in features are imputed using the mean. Rows with missing target values are dropped.

## Customization

-   **Model Parameters**: XGBoost parameters (`XGB_PARAMS_CLASSIFICATION`, `XGB_PARAMS_REGRESSION`), number of walk-forward splits (`N_SPLITS_WALK_FORWARD`), and initial training size ratio (`INITIAL_TRAIN_SIZE_RATIO`) can be adjusted in `xgboost_model.py`.
-   **Feature Set**: Modify the `load_and_preprocess_data` function if you need to change how features are selected or engineered.
-   **Target Variable**: The script runs both classification and regression. You can comment out one of the sections in the `main()` function if you only need one type of model.

## Notes

-   The dataset size is noted as limited (approx. 5 years). The walk-forward validation and regularization techniques are crucial for building a robust model.
-   Ensure your `FeatureData.csv` is correctly formatted and contains sufficient data for meaningful training and validation.
