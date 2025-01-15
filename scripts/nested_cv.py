from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np


SPLIT_POINTS = {
    0: 19716,  # Video 0
    1: 19719,  # Video 1
    2: 19432, # Video 2 
}


def partition_feature_df(feature_df, grp_by, split_points = SPLIT_POINTS):
    """
    Partitions the feature dataframe into folds based on pre-defined split points for each video.

    Parameters:
    - feature_df: pd.DataFrame, the dataframe containing features and episode information.
    - split_points: dict, mapping of video indices to the frame numbers at which to split episodes.

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
    feature_df.groupby(['video_idx', 'fold'])[grp_by]
    .apply(lambda group: group.eq(1).sum())  # Count occurrences where the value equals 1
    .reset_index()
    )

    return feature_df, split_overview



from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

def nested_cross_validation(feature_df, train_cols, target_col, model_class, param_grid, num_cores = -1, scale = True, scoring = 'roc_auc'):
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
        #print(f"Outer Fold: {outer_fold}")

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
            #print(f"  Inner Validation Fold: {inner_val_fold}")

            # Split data into inner training and validation sets
            inner_train_df = outer_train_df[outer_train_df['fold'] != inner_val_fold]
            inner_val_df = outer_train_df[outer_train_df['fold'] == inner_val_fold]

            X_inner_train = inner_train_df[train_cols].values
            y_inner_train = inner_train_df[target_col].values

            X_inner_val = inner_val_df[train_cols].values
            y_inner_val = inner_val_df[target_col].values

            if(scale):
                # Create a pipeline with StandardScaler and the model
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model_class())])
            else:
                pipeline = Pipeline([
                    ('model', model_class())])


            # Update param_grid to reference 'model__'
            pipeline_param_grid = {f'model__{key}': value for key, value in param_grid.items()}

            # Perform grid search on inner folds
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=pipeline_param_grid,
                scoring= scoring,  # Change this metric based on the problem
                cv=[(np.arange(len(X_inner_train)), np.arange(len(X_inner_val)))],
                n_jobs= num_cores
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




import xgboost as xgb # type: ignore
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

def ncv_xgb_gpu(feature_df, train_cols, target_col, param_grid):
    """
    Perform nested cross-validation using GPU-accelerated XGBoost with predefined folds,
    returning models, feature_df, and a summary of results.
    """
    best_models = {}  # Dictionary to store best models for each fold
    results = []      # List to store performance metrics for each fold
    auc_scores = {}   # To store AUC scores for each outer fold
    outer_folds = feature_df['fold'].unique()

    for outer_fold in outer_folds:
        print(f"Outer Fold: {outer_fold}")
        # Outer loop: Split into training and validation sets
        outer_train_df = feature_df[feature_df['fold'] != outer_fold]
        outer_val_df = feature_df[feature_df['fold'] == outer_fold]

        X_outer_train = outer_train_df[train_cols].values
        y_outer_train = outer_train_df[target_col].values
        X_outer_val = outer_val_df[train_cols].values
        y_outer_val = outer_val_df[target_col].values

        inner_folds = outer_train_df['fold'].unique()
        best_model_params = None
        best_score = -np.inf

        # Inner loop for hyperparameter tuning
        for i, params in enumerate(ParameterGrid(param_grid), start=1):
            print(f"Model: {i}/{len(ParameterGrid(param_grid))}", end='\r')
            
            fold_scores = []
            for inner_val_fold in inner_folds:
                # Split inner training and validation sets
                inner_train_df = outer_train_df[outer_train_df['fold'] != inner_val_fold]
                inner_val_df = outer_train_df[outer_train_df['fold'] == inner_val_fold]

                X_inner_train = inner_train_df[train_cols].values
                y_inner_train = inner_train_df[target_col].values
                X_inner_val = inner_val_df[train_cols].values
                y_inner_val = inner_val_df[target_col].values

                # Train and validate model using XGBoost
                dtrain = xgb.DMatrix(X_inner_train, label=y_inner_train)
                dval = xgb.DMatrix(X_inner_val, label=y_inner_val)

                params['tree_method'] = 'hist'  # Use histogram-based tree method
                params['device'] = 'cuda'      # Enable GPU acceleration
                params['objective'] = 'binary:logistic'  # For binary classification
                params['eval_metric'] = 'auc'

                model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, "validation")], verbose_eval=False)
                predictions = model.predict(dval)
                inner_score = roc_auc_score(y_inner_val, predictions)
                fold_scores.append(inner_score)

            # Average inner validation scores
            avg_inner_score = np.mean(fold_scores)

            if avg_inner_score > best_score:
                best_score = avg_inner_score
                best_model_params = params

        # Train the best model on the entire outer training set
        dtrain_outer = xgb.DMatrix(X_outer_train, label=y_outer_train)
        dval_outer = xgb.DMatrix(X_outer_val, label=y_outer_val)
        best_model_params['tree_method'] = 'hist'
        best_model_params['device'] = 'cuda'
        best_model_params['objective'] = 'binary:logistic'
        best_model_params['eval_metric'] = 'auc'

        model = xgb.train(best_model_params, dtrain_outer, num_boost_round=100)
        best_models[outer_fold] = model

        # Evaluate the model on the outer validation set
        predictions = model.predict(dval_outer)
        y_outer_pred = (predictions > 0.5).astype(int)

        # Collect metrics
        metrics = {
            'outer_fold': outer_fold,
            'accuracy': accuracy_score(y_outer_val, y_outer_pred),
            'precision': precision_score(y_outer_val, y_outer_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_outer_val, y_outer_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_outer_val, y_outer_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_outer_val, predictions)
        }

        results.append(metrics)

        # Store AUC for the outer fold
        auc_scores[outer_fold] = metrics['roc_auc']

    # Combine results
    results_df = pd.DataFrame(results)
    summary = results_df.select_dtypes(include=[np.number]).mean().to_dict()
    
    print("\nSummary of Metrics Across Folds:")
    print(summary)
    
    # Return results compatible with evaluate_model
    return results_df, summary, best_models




from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

def evaluate_model(best_models, feature_df, train_cols, target_col, cv_results, spec_fold=None):
    """
    Evaluate a model with performance metrics, confusion matrices, and feature importance.

    Parameters:
    - best_models (dict): Dictionary of models keyed by fold.
    - feature_df (DataFrame): DataFrame containing features, target, and fold information.
    - train_cols (list): List of training feature column names.
    - target_col (str): Name of the target column.
    - cv_results (DataFrame): Cross-validation results containing metrics for each fold.
    - spec_fold (str): Optional. 'best' to evaluate the best fold, 'worst' to evaluate the worst fold.
    """
    # Extract ROC AUC scores from cv_results
    auc_scores = cv_results.set_index('outer_fold')['roc_auc'].to_dict()

    # If only_best_auc is True, find the fold with the highest AUC
    if spec_fold == 'best':
        best_fold = max(auc_scores, key=auc_scores.get)
        best_models = {best_fold: best_models[best_fold]}

    if spec_fold == 'worst':
        best_fold = min(auc_scores, key=auc_scores.get)
        best_models = {best_fold: best_models[best_fold]}

    # Iterate over the specified folds (either all or just the best fold)
    for fold, model in best_models.items():
        # Extract validation data for the fold
        val_data = feature_df[feature_df['fold'] == fold]
        X_val = val_data[train_cols].values
        y_val = val_data[target_col].values

        # Convert X_val to DMatrix if model is a Booster
        if isinstance(model, xgb.Booster):
            X_val = xgb.DMatrix(X_val)

        # Make predictions
        if isinstance(model, xgb.Booster):
            y_pred = (model.predict(X_val) > 0.5).astype(int)  # Threshold for binary classification
            y_proba = model.predict(X_val)
        else:
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        # Print classification report
        print(f"Fold {fold} Classification Report:")
        print(classification_report(y_val, y_pred))

        # Generate confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"],
                    yticklabels=["Class 0", "Class 1"])
        plt.title(f"Confusion Matrix for Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Plot ROC curve if probabilities are available
        if y_proba is not None:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for Fold {fold}')
            plt.legend(loc="lower right")
            plt.show()

        # Plot feature importance if the model is a RandomForestClassifier or has feature_importances_
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            sorted_idx = feature_importance.argsort()[::-1]  # Sort in descending order
            sorted_features = [train_cols[i] for i in sorted_idx]

            plt.figure(figsize=(10, 8))
            sns.barplot(
                x=feature_importance[sorted_idx],
                y=sorted_features,
                palette='viridis'
            )
            plt.title(f"Feature Importance for Fold {fold}")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.tight_layout()
            plt.show()
