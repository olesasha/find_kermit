from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np


SPLIT_POINTS = {
    0: 19716,  # Video 0
    1: 19719,  # Video 1
    2: 19432, # Video 2 
}


def partition_feature_df(feature_df, target_list, split_points = SPLIT_POINTS):
    """
    Partitions the feature dataframe into folds based on pre-defined split points for each video.

    Parameters:
    - feature_df: pd.DataFrame, the dataframe containing features and episode information.
    - split_points: dict, mapping of video indices to the frame numbers at which to split episodes.
    - target_list: list of str of target columns in the annotations
    Returns:
    - feature_df: pd.DataFrame, updated with a new 'fold' column indicating the fold assignment.
    """
    feature_df = feature_df.copy()
    feature_df['fold'] = ""

    # Update the fold values based on the split points
    for video_idx, split_point in split_points.items():
        # Create the "-A" mask for the current video
        mask = feature_df['video_idx'] == video_idx
        
        # Assign the fold value as "{video_idx}-B" for frames after the split point
        feature_df.loc[mask & (feature_df['frame_idx'] < split_point), 'fold'] = f"{video_idx}-A"
        feature_df.loc[mask & (feature_df['frame_idx'] >= split_point), 'fold'] = f"{video_idx}-B"


    split_overview = (
    feature_df.groupby(['video_idx', 'fold'])[target_list]
    .apply(lambda group: group.eq(1).sum())  # Count occurrences where the value equals 1
    .reset_index()
    )

    return feature_df, split_overview



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

def nested_cross_validation(feature_df, train_cols, target_col, model_class, param_grid):
    """
    Perform nested cross-validation for hyperparameter tuning and model evaluation using predefined folds.

    Parameters:
    - feature_df: pd.DataFrame, the dataframe containing features, target variable, and folds.
    - train_cols: list of str, the column names of the training features.
    - target_col: str, the column name of the target variable.
    - model_class: sklearn model class, the machine learning model class (e.g., KNeighborsClassifier).
    - param_grid: dict, parameter grid for hyperparameter tuning.

    Returns:
    - results: pd.DataFrame, containing metrics for each outer fold.
    - summary: dict, summary of metrics across all outer folds.
    - best_models: dict, the best model for each outer fold.
    """
    results = []
    best_models = {}
    outer_folds = feature_df['fold'].unique()

    for outer_fold in outer_folds:
        print(f"Outer Fold: {outer_fold}")

        # Split data into outer training and validation sets
        outer_train_df = feature_df[feature_df['fold'] != outer_fold]
        outer_val_df = feature_df[feature_df['fold'] == outer_fold]

        X_outer_train = outer_train_df[train_cols].values
        y_outer_train = outer_train_df[target_col].values

        X_outer_val = outer_val_df[train_cols].values
        y_outer_val = outer_val_df[target_col].values

        inner_folds = outer_train_df['fold'].unique()

        best_model = None
        best_score = -np.inf

        # Inner Loop for hyperparameter tuning
        for inner_val_fold in inner_folds:
            print(f"  Inner Validation Fold: {inner_val_fold}")

            # Split data into inner training and validation sets
            inner_train_df = outer_train_df[outer_train_df['fold'] != inner_val_fold]
            inner_val_df = outer_train_df[outer_train_df['fold'] == inner_val_fold]

            X_inner_train = inner_train_df[train_cols].values
            y_inner_train = inner_train_df[target_col].values

            X_inner_val = inner_val_df[train_cols].values
            y_inner_val = inner_val_df[target_col].values

            # Perform grid search on inner folds
            grid_search = GridSearchCV(
                estimator=model_class(),
                param_grid=param_grid,
                scoring='f1',  # Change this metric based on the problem
                cv=[(np.arange(len(X_inner_train)), np.arange(len(X_inner_val)))],
                n_jobs=-1
            )

            grid_search.fit(X_inner_train, y_inner_train)

            # Evaluate on the inner validation set
            inner_score = grid_search.best_score_

            if inner_score > best_score:
                best_score = inner_score
                best_model = grid_search.best_estimator_

        # Train the best model on the entire outer training set
        best_model.fit(X_outer_train, y_outer_train)

        # Save the best model for each outer fold
        best_models[outer_fold] = best_model

        # Evaluate on the outer validation set
        y_outer_pred = best_model.predict(X_outer_val)
        y_outer_proba = best_model.predict_proba(X_outer_val)[:, 1] if hasattr(best_model, 'predict_proba') else None

        # Collect metrics
        fold_metrics = {
            'outer_fold': outer_fold,
            'accuracy': accuracy_score(y_outer_val, y_outer_pred),
            'precision': precision_score(y_outer_val, y_outer_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_outer_val, y_outer_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_outer_val, y_outer_pred, average='weighted', zero_division=0)
        }

        if y_outer_proba is not None:
            fold_metrics['roc_auc'] = roc_auc_score(y_outer_val, y_outer_proba)

        print(f"Metrics for Fold {outer_fold}: {fold_metrics}")

        results.append(fold_metrics)

    # Combine results
    results_df = pd.DataFrame(results)

    # Summarize metrics across outer folds
    summary = results_df.select_dtypes(include=np.number).mean().to_dict()
    print("\nOverall Summary:")
    print(summary)

    return results_df, summary, best_models
