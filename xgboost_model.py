# XGBoost Model for Daily to Weekly Bias Forecasting

import pandas as pd
import xgboost as xgb
from xgboost import DMatrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Configuration
FILE_PATH = 'FeatureData.csv'
TARGET_COLUMN_DIRECTION = 'Y_Direction'  # For classification
TARGET_COLUMN_RETURN = 'Y_Return5'    # For regression
DATETIME_COLUMN = 'DateTime'

# Model parameters (can be tuned)
XGB_PARAMS_CLASSIFICATION = {
    # core parameters
    'objective':            'binary:logistic',
    'eval_metric':          'logloss',
    'eta':                  0.05,
    'max_depth':            5,
    'subsample':            0.7,
    'colsample_bytree':     0.7,
    'min_child_weight':     1,
    'gamma':                0.1,
    'lambda':               1,
    'alpha':                0.1,

    # booster control
    'n_estimators':         1000,  # how many rounds you're willing to try
    'early_stopping_rounds': 10,   # centralized early‑stop

    # reproducibility
    'seed':                 42
}

XGB_PARAMS_REGRESSION = {
    'objective':            'reg:squarederror',
    'eval_metric':          'rmse',
    'eta':                  0.05,
    'max_depth':            5,
    'subsample':            0.7,
    'colsample_bytree':     0.7,
    'min_child_weight':     1,
    'gamma':                0.1,
    'lambda':               1,
    'alpha':                0.1,

    'n_estimators':         1000,
    'early_stopping_rounds': 10,

    'seed':                 42
}

N_SPLITS_WALK_FORWARD = 5  # Number of splits for walk-forward validation
INITIAL_TRAIN_SIZE_RATIO = 0.7 # Initial proportion of data for training in walk-forward

def load_and_preprocess_data(file_path, target_column, datetime_column):
    """Loads data from CSV and preprocesses it."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None

    if datetime_column in df.columns:
        # Attempt to parse datetime, but don't fail if it's already suitable or not strictly needed for XGBoost features
        try:
            df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not parse {datetime_column} as datetime: {e}. It will be dropped if not used as index.")
    
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        return None, None

    # Features: all columns except the datetime column and the two potential target columns
    potential_targets = [TARGET_COLUMN_DIRECTION, TARGET_COLUMN_RETURN]
    feature_columns = [col for col in df.columns if col not in [datetime_column] + potential_targets]
    
    X = df[feature_columns]
    y = df[target_column]

    # Ensure no NaN values in features or target (simple imputation or drop)
    # For simplicity, we'll drop rows with NaNs. More sophisticated imputation could be used.
    X = X.fillna(X.mean()) # Impute with mean for numeric features
    if X.isnull().any().any():
        print("Warning: NaN values still present in features after mean imputation. Dropping rows with NaNs.")
        combined = pd.concat([X, y], axis=1)
        combined.dropna(inplace=True)
        X = combined[feature_columns]
        y = combined[target_column]

    if y.isnull().any():
        print("Warning: NaN values found in target. Dropping rows with NaNs in target.")
        combined = pd.concat([X, y], axis=1)
        combined.dropna(subset=[target_column], inplace=True)
        X = combined[feature_columns]
        y = combined[target_column]
        
    print(f"Data loaded: {len(X)} samples, {len(X.columns)} features.")
    print(f"Target variable: {target_column}")
    return X, y

def train_and_evaluate_xgboost(X, y, model_params, n_splits, initial_train_size_ratio, model_type='classification'):
    """Trains and evaluates XGBoost model using walk-forward validation."""
    if X is None or y is None or X.empty or y.empty:
        print("Error: Input data (X or y) is missing or empty.")
        return

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Adjusting TimeSeriesSplit to better simulate walk-forward with expanding window
    # The default TimeSeriesSplit might not be ideal for small datasets if initial splits are too small.
    # We'll implement a more controlled walk-forward split.
    
    all_preds = []
    all_true = []
    
    total_samples = len(X)
    # Determine initial training size
    initial_train_samples = int(total_samples * initial_train_size_ratio)
    if initial_train_samples < n_splits : # Ensure enough samples for initial train + subsequent test sets
        print(f"Warning: Initial train size ({initial_train_samples}) is very small relative to n_splits ({n_splits}). Adjusting.")
        initial_train_samples = n_splits * 2 # A heuristic
        if initial_train_samples >= total_samples - n_splits:
             print("Error: Dataset too small for specified n_splits and initial_train_size_ratio.")
             return

    test_set_size = (total_samples - initial_train_samples) // n_splits
    if test_set_size == 0 and n_splits > 0:
        print("Warning: Test set size is 0. Reducing n_splits or initial_train_size_ratio might be needed.")
        # Fallback to a single split if walk-forward cannot be meaningfully applied
        if total_samples * (1-initial_train_size_ratio) > 0:
            test_set_size = int(total_samples * (1-initial_train_size_ratio))
            n_splits = 1
            print(f"Adjusting to a single split with test size: {test_set_size}")
        else:
            print("Error: Not enough data for even a single test split with current initial_train_size_ratio.")
            return
    elif test_set_size == 0 and n_splits == 0:
        print("Error: n_splits is 0, cannot perform validation.")
        return

    print(f"Walk-forward validation: {n_splits} splits.")
    print(f"Initial training samples: {initial_train_samples}")
    print(f"Test set samples per split: {test_set_size}")

    for i in range(n_splits):
        train_end_idx = initial_train_samples + i * test_set_size
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_set_size

        if test_end_idx > total_samples:
            test_end_idx = total_samples # Ensure we don't go out of bounds on the last split
        if test_start_idx >= test_end_idx: # Skip if test set is empty
            print(f"Skipping split {i+1} due to empty test set.")
            continue

        X_train, X_test = X.iloc[:train_end_idx], X.iloc[test_start_idx:test_end_idx]
        y_train, y_test = y.iloc[:train_end_idx], y.iloc[test_start_idx:test_end_idx]
        
        if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
            print(f"Skipping split {i+1} due to empty train/test data after slicing.")
            continue

        print(f"Split {i+1}/{n_splits}: Train size={len(X_train)}, Test size={len(X_test)}")

        if model_type == 'classification':
            model = xgb.XGBClassifier(**model_params)
        elif model_type == 'regression':
            model = xgb.XGBRegressor(**model_params)
        else:
            raise ValueError("model_type must be 'classification' or 'regression'")

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        # eval_metric is already defined in model_params and should be used by fit() when eval_set is present.
        
        # If save_best=True, the model is already the best one. 
        # If you need to load the best model explicitly (e.g. if save_best=False and you track best_iteration):
        # if hasattr(model, 'best_iteration') and model.best_iteration is not None:
        #     preds = model.predict(X_test, iteration_range=(0, model.best_iteration + 1))
        # else:
        #     preds = model.predict(X_test)
        preds = model.predict(X_test)
        all_preds.extend(preds)
        all_true.extend(y_test.values)

        # Feature importance for the last fold (or could average them)
        if i == n_splits - 1:
            try:
                feature_importances = pd.Series(model.feature_importances_, index=X.columns)
                print("\nFeature Importances (last fold):")
                print(feature_importances.sort_values(ascending=False).head(10))
                # Plot feature importances
                plt.figure(figsize=(10, 6))
                feature_importances.sort_values(ascending=False).head(15).plot(kind='barh')
                plt.title(f'Top 15 Feature Importances ({model_type} - Last Fold)')
                plt.tight_layout()
                plt.savefig(f'feature_importances_{model_type}.png')
                print(f"Saved feature importance plot to feature_importances_{model_type}.png")
                plt.close()
            except Exception as e:
                print(f"Could not plot/print feature importances: {e}")

    if not all_true or not all_preds:
        print("No predictions were made. Check data and split logic.")
        return

    print("\nOverall Walk-Forward Validation Results:")
    if model_type == 'classification':
        accuracy = accuracy_score(all_true, all_preds)
        f1 = f1_score(all_true, all_preds, average='weighted') # Use 'binary' if only two classes and want specific metrics
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score (weighted): {f1:.4f}")
        # Confusion Matrix could be added here
    elif model_type == 'regression':
        rmse = np.sqrt(mean_squared_error(all_true, all_preds))
        r2 = r2_score(all_true, all_preds)
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R-squared: {r2:.4f}")

def main():
    print("--- Running XGBoost Model for Directional Bias (Classification) ---")
    X_class, y_class = load_and_preprocess_data(FILE_PATH, TARGET_COLUMN_DIRECTION, DATETIME_COLUMN)
    if X_class is not None and y_class is not None:
        # Ensure target is integer type for classification if it's not already
        if not pd.api.types.is_integer_dtype(y_class) and pd.api.types.is_numeric_dtype(y_class):
            print(f"Converting target {TARGET_COLUMN_DIRECTION} to integer for classification.")
            y_class = y_class.astype(int)
        elif not pd.api.types.is_numeric_dtype(y_class):
            print(f"Error: Target column {TARGET_COLUMN_DIRECTION} is not numeric and cannot be converted to int.")
            return
        train_and_evaluate_xgboost(X_class, y_class, XGB_PARAMS_CLASSIFICATION, N_SPLITS_WALK_FORWARD, INITIAL_TRAIN_SIZE_RATIO, model_type='classification')
    else:
        print("Skipping classification model due to data loading/preprocessing issues.")

    print("\n--- Running XGBoost Model for Return Prediction (Regression) ---")
    X_reg, y_reg = load_and_preprocess_data(FILE_PATH, TARGET_COLUMN_RETURN, DATETIME_COLUMN)
    if X_reg is not None and y_reg is not None:
        if not pd.api.types.is_numeric_dtype(y_reg):
            print(f"Error: Target column {TARGET_COLUMN_RETURN} is not numeric.")
            return
        train_and_evaluate_xgboost(X_reg, y_reg, XGB_PARAMS_REGRESSION, N_SPLITS_WALK_FORWARD, INITIAL_TRAIN_SIZE_RATIO, model_type='regression')
    else:
        print("Skipping regression model due to data loading/preprocessing issues.")

if __name__ == '__main__':
    main()