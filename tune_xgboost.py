import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import time
from datetime import datetime

TRAIN_PATH = 'dataset/train.csv'
TEST_PATH = 'dataset/test.csv'
SUBMISSION_PATH = 'dataset/submission_xgboost_tuned.csv'
N_TRIALS = 50  
USE_GPU = True  

def check_gpu_support():
    """Check if GPU support is available for XGBoost."""
    try:
        test_data = xgb.DMatrix(np.random.rand(100, 10), label=np.random.randint(0, 2, 100))
        test_params = {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda'}
        xgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
        print("✓ GPU support detected and working!")
        return True
    except Exception as e:
        print(f"✗ GPU support not available: {str(e)}")
        print("  Falling back to CPU training...")
        return False

def get_base_params(use_gpu=False):
    """Get base parameters for XGBoost with optional GPU support."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',  
        'random_state': 42,
    }
    
    if use_gpu:
        params['device'] = 'cuda'  
    else:
        params['device'] = 'cpu'
        params['n_jobs'] = -1
    
    return params
def load_and_prep_data(train_path, test_path):
    print("Loading and preparing data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_ids = test_df['id']
    test_df['y'] = np.nan
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    binary_map = {'yes': 1, 'no': 0}
    for col in ['default', 'housing', 'loan']:
        if col in full_df.columns:
            full_df[col] = full_df[col].map(binary_map)
    
    if 'pdays' in full_df.columns:
        full_df['was_previously_contacted'] = (full_df['pdays'] != -1).astype(int)
    
    # Handle categorical columns - encode them properly for XGBoost
    categorical_columns = []
    for col in full_df.select_dtypes(include='object').columns:
        if col not in ['id']:  
            full_df[col] = pd.Categorical(full_df[col]).codes
            categorical_columns.append(col)
    
    print(f"Encoded categorical columns: {categorical_columns}")
        
    return full_df, test_ids

def objective(trial, X, y):
    """The function Optuna will optimize."""
    
    # Define the hyperparameter search space
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'random_state': 42,
        'n_jobs': -1,
        'enable_categorical': True
    }
    
    # Perform cross-validation with the given parameters
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        
        # Simple fit without early stopping for compatibility
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)[:, 1]
        fold_scores.append(roc_auc_score(y_val, preds))
        
        # Clean up memory
        del model
        gc.collect()
        
    # Return the mean score, which Optuna will try to maximize
    return np.mean(fold_scores)

def objective_with_early_stopping(trial, X, y, use_gpu=False):
    """Alternative objective function with manual early stopping using validation split."""
    
    trial_start_time = time.time()
    
    # Get base parameters (CPU or GPU)
    base_params = get_base_params(use_gpu)
    
    # Add hyperparameters to optimize
    base_params.update({
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 3),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
    })
    
    # Perform cross-validation with the given parameters
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    print(f"  Trial {trial.number + 1}: Testing hyperparameters...")
    for key, value in base_params.items():
        if key not in ['objective', 'eval_metric', 'random_state', 'n_jobs', 'tree_method', 'device']:
            print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"    Fold {fold_idx + 1}/5: ", end="", flush=True)
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Use native XGBoost with DMatrix for better control
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train with early stopping and reduced rounds for faster tuning
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(
            base_params,
            dtrain,
            num_boost_round=300,  
            evals=evallist,
            early_stopping_rounds=20,  
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        fold_auc = roc_auc_score(y_val, preds)
        fold_scores.append(fold_auc)
        
        fold_time = time.time() - fold_start_time
        print(f"AUC: {fold_auc:.4f} ({fold_time:.1f}s)")
        
        # Clean up memory
        del model, dtrain, dval
        gc.collect()
    
    trial_time = time.time() - trial_start_time
    mean_auc = np.mean(fold_scores)
    print(f"  Trial {trial.number + 1} Complete: Mean AUC: {mean_auc:.4f} ± {np.std(fold_scores):.4f} ({trial_time:.1f}s)")
        
    return mean_auc

if __name__ == "__main__":
    # Check XGBoost version and GPU support
    print(f"XGBoost version: {xgb.__version__}")
    gpu_available = check_gpu_support() if USE_GPU else False
    
    if USE_GPU and not gpu_available:
        print("GPU requested but not available. Using CPU instead.")
        USE_GPU = False
    
    print(f"Training mode: {'GPU' if gpu_available and USE_GPU else 'CPU'}")
    print("=" * 50)
    
    full_df, test_ids = load_and_prep_data(TRAIN_PATH, TEST_PATH)
    
    train_df = full_df[full_df['y'].notna()]
    test_df = full_df[full_df['y'].isna()]
    
    features = [col for col in train_df.columns if col not in ['id', 'y']]
    X = train_df[features]
    y = train_df['y']
    X_test = test_df[features]

    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Features: {features}")

    #  Run Optuna Study
    print(f"\nStarting hyperparameter optimization with {N_TRIALS} trials...")
    print("Using native XGBoost with early stopping for better performance...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Create study with progress callback
    study = optuna.create_study(direction='maximize')
    
    def progress_callback(study, trial):
        elapsed_time = time.time() - start_time
        trials_completed = len(study.trials)
        
        if trials_completed > 0:
            avg_time_per_trial = elapsed_time / trials_completed
            estimated_remaining_time = avg_time_per_trial * (N_TRIALS - trials_completed)
            
            print(f"\n--- Progress Report ---")
            print(f"Trials completed: {trials_completed}/{N_TRIALS}")
            print(f"Best AUC so far: {study.best_value:.6f}")
            print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
            print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
            print(f"Progress: {trials_completed/N_TRIALS*100:.1f}%")
            print("=" * 80)
    
    study.optimize(
        lambda trial: objective_with_early_stopping(trial, X, y, use_gpu=gpu_available and USE_GPU), 
        n_trials=N_TRIALS,
        callbacks=[progress_callback]
    )

    print("\n" + "=" * 80)
    print("--- Optuna Study Complete ---")
    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time/60:.1f} minutes")
    print(f"Average time per trial: {total_time/N_TRIALS:.1f} seconds")
    print(f"Best Trial AUC: {study.best_value:.6f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    print("\nRetraining with best hyperparameters on full training data...")
    best_params = study.best_params.copy()
    
    final_base_params = get_base_params(gpu_available and USE_GPU)
    best_params.update(final_base_params)

    # We retrain using cross-validation again for a robust test prediction
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    
    print(f"Training {skf.n_splits} models for ensemble prediction...")
    final_start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print(f"Training Fold {fold+1}/{skf.n_splits}...", end=" ", flush=True)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Use native XGBoost for final training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)
        
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        oof_preds[val_idx] = model.predict(dval)
        
        test_preds += model.predict(dtest) / skf.n_splits
        
        # Print fold performance
        fold_auc = roc_auc_score(y_val, model.predict(dval))
        fold_time = time.time() - fold_start_time
        print(f"AUC: {fold_auc:.6f} ({fold_time:.1f}s)")
        
        del model, dtrain, dval, dtest
        gc.collect()
    
    final_time = time.time() - final_start_time
    print(f"\nFinal training completed in {final_time:.1f} seconds")
    
    # Calculate overall CV score
    cv_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall CV AUC: {cv_auc:.6f}")
    
    # --- Generate Submission File ---
    submission_df = pd.DataFrame({'id': test_ids, 'y': test_preds})
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nTuned XGBoost submission saved to {SUBMISSION_PATH}")
    print("Submission preview:")
    print(submission_df.head(10))
    print(f"\nPrediction statistics:")
    print(f"  Min: {test_preds.min():.6f}")
    print(f"  Max: {test_preds.max():.6f}")
    print(f"  Mean: {test_preds.mean():.6f}")
    print(f"  Std: {test_preds.std():.6f}")