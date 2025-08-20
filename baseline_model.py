import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import gc

TRAIN_PATH = "dataset/train.csv"
TEST_PATH = "dataset/test.csv"
SUBMISSION_PATH = "dataset/my_sample_submission.csv"

# Load data
def load_data(train_path, test_path):
    """Loads the training and test datasets."""
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df

# Create baseline model and preprocessing pipeline
def create_baseline_pipeline(train_df):
    """Identifies feature types and creates a preprocessing and modeling pipeline"""
    print("Creating baseline model pipeline...")

    # Identify numerics and categorical features
    numeric_features = train_df.select_dtypes(include=np.number).columns.tolist()
    numeric_features.remove('id')
    numeric_features.remove('y')

    categorical_features = train_df.select_dtypes(include='object').columns.tolist()

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    # Categorical features will be one-hot encoded.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any), though we've covered all
    )

    # Create the full pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    return pipeline, numeric_features + categorical_features

def train_and_evaluate(pipeline, X, y):
    """Trains the model using stratified k-fold cross-validation and evaluates it."""
    print("Training and evaluating model with Stratified K-Fold CV...")
    
    # Using StratifiedKFold because of the class imbalance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/5 ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Fit the pipeline on the training data for this fold
        pipeline.fit(X_train, y_train)

        # Predict probabilities on the validation set
        val_preds = pipeline.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds

        # Calculate and store the AUC score for the fold
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}")
        
        # Clean up memory
        del X_train, y_train, X_val, y_val
        gc.collect()

    # Overall Out-of-Fold (OOF) AUC score
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall OOF AUC: {overall_auc:.5f}")
    print(f"Mean Fold AUC: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")

    return pipeline

def generate_submission(pipeline, test_df, features, submission_path):
    """Generates predictions on the test set and creates a submission file."""
    print("Generating submission file...")

    # Predict probabilities on the test data
    test_predictions = pipeline.predict_proba(test_df[features])[:, 1]

    # Create submission DataFrame
    submission_df = pd.DataFrame({'id': test_df['id'], 'y': test_predictions})
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission file saved to {submission_path}")
    print(submission_df.head())


if __name__ == "__main__":
    # Load data
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # Define features (X) and target (y)
    X = train_df.drop(columns=['id', 'y'])
    y = train_df['y']

    # Create the model pipeline
    pipeline, feature_names = create_baseline_pipeline(train_df)

    print("\n--- Training on full dataset for final prediction ---")
    pipeline.fit(X, y)
    
    # Generate the submission file
    generate_submission(pipeline, test_df, feature_names, SUBMISSION_PATH)