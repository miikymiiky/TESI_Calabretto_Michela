#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 15:57:11 2025

@author: michela
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
import json

# Imports for ML Models and Evaluation 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Evaluation Metrics
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score)

# Imports for Cross-Validation 
from sklearn.model_selection import GroupKFold, cross_validate, GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")



class ModelTrainer:
    """
    A class to handle the training of machine learning models and deep learning algorithms for 
    predicting postprandial glycemic events.
    """
    def __init__(self, data_file_path: str, features_file_path: str, analysis_type: str, 
                 classification_target: str, regression_target: str,
                 test_size: float = 0.2, random_state: int = 42, n_splits: int = 5):
        """
        Initializes the ModelTrainer.

        Args:
            data_file_path (str): Path to the CSV file with features.
            features_file_path (str): Path to the JSON file with selected feature lists.
            analysis_type (str): Mode of analysis, 'classification' or 'regression'.
            classification_target (str): Target column for classification.
            regression_target (str): Target column for regression.
            test_size (float): The proportion of the dataset to allocate to the test set.
            random_state (int): Seed for reproducibility of the random split.
            n_splits (int): Number of folds for cross validation.
        """
        self.data_file_path = data_file_path
        self.features_file_path = features_file_path
        self.analysis_type = analysis_type
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = n_splits
        
        self.classification_target = classification_target
        self.regression_target = regression_target

        if 'classification' in self.analysis_type:
            self.target_column = classification_target
        elif self.analysis_type == 'regression':
            self.target_column = regression_target
        else:
            raise ValueError("analysis_type must contain 'classification' or be 'regression'")


        self.df = self._load_data()
        self.feature_sets = self._load_feature_sets()
        self.results_log = []


    def _load_data(self) -> pd.DataFrame:
        """Loads and prepares the feature DataFrame."""
        if not os.path.exists(self.data_file_path):
            print(f"Error: Data file not found at '{self.data_file_path}'")
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.data_file_path)
            # Ensure datetime columns are correctly parsed if they are needed later
            for col in ['meal_time_start', 'meal_time_end']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Successfully loaded data from {self.data_file_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    
    def _load_feature_sets(self) -> dict:
        """Loads the pre-selected feature sets from a JSON file."""
        print(f"\n--- Loading Pre-Selected Feature Sets from: {self.features_file_path} ---")
        try:
            with open(self.features_file_path, 'r') as f:
                feature_sets = json.load(f)
            print("Successfully loaded feature sets.")
            for name, features in feature_sets.items():
                print(f"  - Found set '{name}' with {len(features)} features.")
            return feature_sets
        except FileNotFoundError:
            print(f"ERROR: Features file not found at '{self.features_file_path}'.")
            print("Please run the data analysis step in main.py to generate it.")
            return {}
        except Exception as e:
            print(f"ERROR: Could not load or parse features file: {e}")
            return {} 
        
            
    
    def _split_data_by_subject(self):
        """
        Splits data into training and test sets using a greedy approach to better
        balance the row counts, while ensuring subjects are unique to each set.
        """
        if self.df.empty or 'subj_id' not in self.df.columns:
            print("Cannot split data. DataFrame is empty or 'subj_id' column is missing.")
            return

        # 1. Calculate the number of meals (rows) per subject
        meal_counts = self.df['subj_id'].value_counts() # meals recorded per subject (meal_counts is a serie where theindex is the subject id and the value is the numbr of meals for that subject)
        subjects = meal_counts.index.tolist()

        # 2. Shuffle the subjects for random selection
        # Use a seeded random instance for reproducibility
        random.Random(self.random_state).shuffle(subjects) # random.Random(self.random_state) ---> creates a new Random instance and seeds it with self.random_state to always get the same result
                                                           # .shuffle() ---> reorders the list subjects in place so that the subjects will be in a randomized order

        # 3. Iteratively build the test set
        target_test_rows = int(len(self.df) * self.test_size)
        test_subjects = []
        current_test_rows = 0

        for subject in subjects:
            subject_rows = meal_counts[subject]
            # Add the subject to the test set if it gets us closer to the target without significantly overshooting it
            if abs(target_test_rows - (current_test_rows + subject_rows)) < abs(target_test_rows - current_test_rows):
                 test_subjects.append(subject)
                 current_test_rows += subject_rows
        
        # 4. The remaining subjects form the training set
        train_subjects = [s for s in self.df['subj_id'].unique() if s not in test_subjects]

        # 5. Create the final DataFrames
        self.train_df = self.df[self.df['subj_id'].isin(train_subjects)].copy()
        self.test_df = self.df[self.df['subj_id'].isin(test_subjects)].copy()
        
        print("\n--- Data Split Summary  ---")
        print(f"Total subjects: {len(subjects)}")
        print(f"Training subjects ({len(train_subjects)})")
        print(f"Test subjects ({len(test_subjects)})")
        print("-" * 25)
        print(f"Total postprandial periods: {len(self.df)}")
        print(f"Training periods: {len(self.train_df)} ({len(self.train_df) / len(self.df):.1%})")
        print(f"Test periods: {len(self.test_df)} ({len(self.test_df) / len(self.df):.1%})")
        print("-" * 25)
    
    
    
    def _get_models(self) -> dict:
        """Defines the models to be trained based on the analysis type."""
        models = {}
        if 'classification' in self.analysis_type:
            # Common models for both classification types
            models["Logistic Regression"] = LogisticRegression(random_state=self.random_state, max_iter=10000, class_weight='balanced') # class_weight='balanced' ---> adjusts weights inversely proportional to class frequencies to avoid biased toward the majority 
            models["Support Vector Machine"] = SVC(random_state=self.random_state, class_weight='balanced', probability=True) # probability=True to also add output probabilities
            models["Random Forest"] = RandomForestClassifier(random_state=self.random_state, class_weight='balanced_subsample', n_jobs=-1) # class_weight='balanced_subsample' ---> for unbalanced datasets: creates balanced class weights for each individual tree based on its bootstrap sample
            models["LightGBM"] = LGBMClassifier(random_state=self.random_state, class_weight='balanced',n_jobs=-1, verbose=-1) # verbose=-1 ---> to silence the prints in the console
            
            # Specific model parameters
            if self.analysis_type == 'binary_classification':
                models["XGBoost"] = XGBClassifier(random_state=self.random_state, eval_metric='logloss', n_jobs=-1) # eval_metric='logloss' ---> logarithmic loss (evaluation metric used for the training)
            elif self.analysis_type == 'multi_class_classification':
                models["XGBoost"] = XGBClassifier(random_state=self.random_state, eval_metric='mlogloss', n_jobs=-1)
                
            # ADD NEW CLASSIFICATION MODELS HERE
        
        elif self.analysis_type == 'regression':
            models["Ridge Regression"] = Ridge(random_state=self.random_state)
            models["Lasso Regression"] = Lasso(random_state=self.random_state, max_iter=5000)
            models["Support Vector Regressor"] = SVR()
            models["Random Forest Regressor"] = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            models["XGBoost Regressor"] = XGBRegressor(random_state=self.random_state, n_jobs=-1)
            models["LightGBM Regressor"] = LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1)
            
            # ADD NEW REGRESSION MODELS HERE
    
            
        return models
    
    
    
    def _get_parameter_grids(self) -> dict:
        """Defines the hyperparameter grids for each model."""
        param_grids = {}
        if 'classification' in self.analysis_type:
            # Common grids for both classification types
            param_grids["Support Vector Machine"] = [
                {
                    'model__kernel': ['linear'], # uses a simple straigth line as a decision boundary
                    'model__C': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                {
                    'model__kernel': ['rbf'], # Radial Basis Function: creates a complex, flexible, curved boundary
                    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'model__gamma': ['scale', 'auto'] # 'model__gamma': defines how much influence a single training example has
                                                      # 'scale' ---> sets the gamma value based on both the number of features and the variance of the data
                                                      # 'auto' ---> sets the gamma value based only on the number of features
                }
            ]
            param_grids["Random Forest"] = {
               'model__n_estimators': [100, 200, 300, 400, 500, 600, 700], # number of trees (400 has way higher precision, but very vewry low recall, while 200 the inverse)
               'model__max_depth': [10, 20, 30, None], # maximum depth of each tree
               'model__min_samples_split': [2, 5, 10], # minimum number of samples to allow a further split of an internal node [i.e. requires at least 5 samples to split]
               'model__min_samples_leaf': [1, 2, 4] # minimum number of samples required to be at a leaf node at the end of the branch 
            }
            param_grids["LightGBM"] = [
                {
                    'model__learning_rate': [0.1],
                    'model__n_estimators': [100, 200],
                    'model__num_leaves': [31, 40],
                    'model__max_depth': [-1, 10]
                },
                {
                    'model__learning_rate': [0.01],
                    'model__n_estimators': [200, 400, 600], # More trees for the lower learning rate
                    'model__num_leaves': [31, 40],
                    'model__max_depth': [-1, 10]
                }
            ]
            
            # XGBoost grid depends on the classification type
            if self.analysis_type == 'binary_classification':
                param_grids["Logistic Regression"] = [
                    {
                        'model__penalty': ['l1'], # Lasso regularization
                        'model__solver': ['liblinear', 'saga'], # model__solver: algorithm used to optimize the logistic regression
                                                                # 'liblinear' ---> good for small datasets; 'saga' ---> scales better to large datasets
                        'model__C': [0.01, 0.1, 1, 10, 100] # model__C: Inverse of regularization strength. Smaller C means stronger regularization
                    },
                    {
                        'model__penalty': ['l2'], # Ridge regularization
                        'model__solver': ['lbfgs', 'liblinear', 'saga'], # 'lbfgs' ---> works well for small to medium datasets, but can also handle larger datasets efficiently
                        'model__C': [0.01, 0.1, 1, 10, 100]
                    }
                ]
                param_grids["XGBoost"] = {
                    'model__n_estimators': [100, 200], # Number of boosting rounds (number of trees)
                    'model__max_depth': [1, 2, 3, 4, 5, 7], # Max depth of each tree
                    'model__learning_rate': [0.01, 0.1], # Step size shrinkage ---> shrinks the contribution of each tree by this factor: smaller values make learning more gradual, often requiring more trees; larger values converge faster but can overshoot
                    'model__scale_pos_weight': [1, 5] # A few manual values for class imbalance
                }
            elif self.analysis_type == 'multi_class_classification':
                print("Using parameter grid without 'liblinear' for Multi-Class Logistic Regression.")
                param_grids["Logistic Regression"] = [
                    {
                        'model__penalty': ['l1'],
                        'model__solver': ['saga'],
                        'model__C': [0.1, 1, 10, 100],
                        'model__max_iter': [10000, 25000]
                    },
                    {
                        'model__penalty': ['l2'],
                        'model__solver': ['lbfgs', 'saga'],
                        'model__C': [0.1, 1, 10, 100]
                    }
                ]
                param_grids["XGBoost"] = {
                    'model__n_estimators': [100, 200, 300, 400],
                    'model__max_depth': [1, 2, 3, 4, 5, 6, 7],
                    'model__learning_rate': [0.01, 0.1],
                    # No scale_pos_weight for multi-class
                    'model__subsample': [0.7, 1.0], # Fraction of training data to use per tree
                    'model__colsample_bytree': [0.7, 1.0] # Fraction of features to use per tree
                }
        
        elif self.analysis_type == 'regression':
            param_grids["Ridge Regression"] = {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 500.0]  # controls the strength of the L2 regularization
            }
            param_grids["Lasso Regression"] = {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0] # L1 regularization strength
            }
            param_grids["Support Vector Regressor"] = [
                {
                    'model__kernel': ['linear'],
                    'model__C': [0.01, 0.1, 1, 10, 100]
                },
                {
                    'model__kernel': ['rbf'],
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__gamma': ['scale', 'auto']
                }
            ]
            param_grids["Random Forest Regressor"] = {
               'model__n_estimators': [100, 200, 300, 400, 500],
               'model__max_depth': [10, 20, None],
               'model__min_samples_split': [2, 5],
               'model__min_samples_leaf': [1, 2]
            }
            param_grids["XGBoost Regressor"] = {
                'model__n_estimators': [100, 200, 300, 400, 500],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1]
            }
            param_grids["LightGBM Regressor"] = {
                'model__learning_rate': [0.01, 0.1],
                'model__n_estimators': [100, 200, 300, 400, 500, 600],
                'model__num_leaves': [20, 31, 40],
            }
        return param_grids
    
    
    
    def _perform_grid_search(self, pipeline, param_grid, X_train, y_train, cv, groups):
        """Performs GridSearchCV to find the best hyperparameters for a model."""
        print("Performing grid search with cross-validation...")
        
        # Ttrack multiple metrics during the search
        if self.analysis_type == 'binary_classification':
            scoring_metrics = ['roc_auc', 'recall', 'average_precision']
            refit_metric = 'roc_auc'
        elif self.analysis_type == 'multi_class_classification':
            scoring_metrics = ['roc_auc_ovr_weighted', 'recall_weighted', 'f1_weighted'] # roc_auc_ovr_weighted ---> 'ovr' = One Versus Rest(each class is compared against all the others); 'weighted' = scores are weighted by how many samples are in each class
            refit_metric = 'roc_auc_ovr_weighted'
        else: # Regression
            # Note: Scikit-learn uses scores where higher is better.
            # So we use 'neg_mean_absolute_error' and the grid search will try to make it closer to 0.
            scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            refit_metric = 'r2'
        
        # Use roc_auc for scoring as it's a good metric for imbalanced classes
        # Use 'refit' to choose the primary metric
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, 
                                   scoring=scoring_metrics, refit=refit_metric,
                                   n_jobs=-1, verbose=1) # n_jobs=-1 ---> use all available CPU cores for faster execution
                                                         # verbose=1 ---> controls how much information is printed while the grid search runs; 1 means it will print basic progress information for each combination being tested
        
        # Fit the grid search to the data. It will automatically handle the CV withouth subject leakage.
        grid_search.fit(X_train, y_train, groups=groups)
        
        print("\n--- Grid Search Results ---")
        print(f"Best Parameters Found (optimized for {refit_metric.upper()}): {grid_search.best_params_}")
        print(f"Best CV {refit_metric.upper()} Score: {grid_search.best_score_:.4f}")
       
        # Return all tuning results 
        tuning_results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Return the best model AND the detailed results dataframe
        return grid_search.best_estimator_, tuning_results_df
    
    
    
    def _evaluate_classification(self, model_name: str, best_params, y_true, y_pred, y_pred_proba):
        """Dispatcher function that calls the appropriate evaluation method."""
        if self.analysis_type == 'binary_classification':
            # For binary, we need the probability of the positive class (class 1)
            return self._evaluate_binary_classification(model_name, best_params, y_true, y_pred, y_pred_proba[:, 1])
        elif self.analysis_type == 'multi_class_classification':
            # For multi-class, we need the full probability matrix
            return self._evaluate_multi_class_classification(model_name, best_params, y_true, y_pred, y_pred_proba)

    
    
    def _evaluate_binary_classification(self, model_name: str, best_params, y_true, y_pred, y_pred_proba):
        """Calculates and prints a comprehensive set of classification metrics."""
        print(f"\n--- Evaluation Metrics for: {model_name} (Binary) ---")

        # Confusion Matrix and related metrics
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) # labels=[0, 1],to fixing the order: 0 is the negative class (No Hypo), 1 is the positive class (Hypo + Deep Hypo) 
        print("Confusion Matrix:")
        print(f"         Predicted 0   Predicted 1")
        print(f"Actual 0 [[{cm[0][0]:^11d} {cm[0][1]:^11d}]] (No Hypo)") # : ^11d ---> d = format as an integer, ^11 = center it in a field of width 11 (to keep the table aligned)
        print(f"Actual 1 [[{cm[1][0]:^11d} {cm[1][1]:^11d}]] (Hypo)")
        print("-" * 40)
        
        tn, fp, fn, tp = cm.ravel() # .ravel() flattens the 2D matrix into a 1D array
                                                                     
        accuracy = accuracy_score(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity_recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity_recall) / (precision + sensitivity_recall) if (precision + sensitivity_recall) > 0 else 0
        
        # Other key metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        # Print report
        print(f"Accuracy:             {accuracy:.4f}")
        print(f"Precision:            {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity_recall:.4f}")
        print(f"Specificity:          {specificity:.4f}")
        print(f"F1 Score:             {f1_score:.4f}")
        print(f"MCC:                  {mcc:.4f}")
        print(f"AUC-ROC:              {roc_auc:.4f}")
        print(f"AUC-PR:               {pr_auc:.4f}")
        print("-" * 40)
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Hypo', 'Hypo'])) # Generates a text report of classification metrics for each class. It computes precision, recall, f1-score, and support
        
        # Create a dictionary of the results
        metrics = {
            'model': model_name,
            'best_params': str(best_params), # Convert to string for CSV
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision,
            'recall': sensitivity_recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'mcc': matthews_corrcoef(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': average_precision_score(y_true, y_pred_proba)
        }
        return metrics
    
    
    
    def _evaluate_multi_class_classification(self, model_name: str, best_params, y_true, y_pred, y_pred_proba):
        """Calculates and prints multi-class classification metrics."""
        print(f"\n--- Evaluation Metrics for: {model_name} (Multi-Class) ---")
        class_labels = ['No Hypo', 'Hypo', 'Deep Hypo']
        target_indices = [0, 1, 2]

        cm = confusion_matrix(y_true, y_pred, labels=target_indices)
        print("Confusion Matrix:")
        print(f"                   Pred_No_Hypo   Pred_Hypo   Pred_Deep_Hypo")
        for i, label in enumerate(class_labels):
            print(f"Actual_{label:<10}  {cm[i]}")
        print("-" * 60)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) # zero_division=0 ---> for when class has no predicted samples (denominator is 0); this way, instead of crashing, sklearn will return 0
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        specificities = {}
        for i in target_indices:
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities[class_labels[i]] = specificity

        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')

        print(f"Accuracy (Overall):        {accuracy:.4f}")
        print(f"Precision (Weighted):      {precision:.4f}")
        print(f"Recall (Weighted):         {recall:.4f}")
        print(f"F1 Score (Weighted):       {f1:.4f}")
        print(f"MCC (Matthews Corr Coef):  {mcc:.4f}")
        print(f"AUC-ROC (OvR, Weighted):   {roc_auc:.4f}")
        print("\nSpecificity (per class):")
        for label, spec in specificities.items():
            print(f"  - {label}: {spec:.4f}")
        print("-" * 60)
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))
        
        metrics = {
            'model': model_name, 'best_params': str(best_params), 'accuracy': accuracy,
            'precision_weighted': precision, 'recall_weighted': recall, 'f1_weighted': f1, 'mcc': mcc,
            'roc_auc_ovr_weighted': roc_auc, 'specificity_no_hypo': specificities.get('No Hypo', 0),
            'specificity_hypo': specificities.get('Hypo', 0), 'specificity_deep_hypo': specificities.get('Deep Hypo', 0)
        }
        return metrics
    
    
    
    def _evaluate_regression(self, model_name: str, best_params, y_true, y_pred):
        """Calculates and prints a comprehensive set of regression metrics."""
        print(f"\n--- Final Evaluation on Test Set: {model_name} ---")
        print(f"Best Parameters Found: {best_params}")
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"Mean Absolute Error (MAE):      {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²):                 {r2:.4f}")

        # Return metrics for logging
        metrics = {
            'model': model_name,
            'best_params': str(best_params),
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        return metrics
    
    
    
    def save_results_to_csv(self, filename="model_performance_log.csv"):
        """Saves the collected experiment results to a CSV file."""
        if not self.results_log:
            print("No results to save.")
            return
        
        results_df = pd.DataFrame(self.results_log)
        
        if self.analysis_type == 'binary_classification':
            column_order = [
                'feature_set', 'model', 'auc_roc', 'recall', 'precision', 'specificity', 
                'f1_score', 'mcc', 'accuracy', 'auc_pr', 'best_params'
            ]
            sort_by_col = 'auc_roc'
        elif self.analysis_type == 'multi_class_classification':
            column_order = [
                'feature_set', 'model', 'roc_auc_ovr_weighted', 'recall_weighted', 'precision_weighted', 
                'f1_weighted', 'mcc', 'accuracy', 'specificity_no_hypo', 'specificity_hypo', 
                'specificity_deep_hypo', 'best_params'
            ]
            sort_by_col = 'roc_auc_ovr_weighted'
        else: # Regression
            column_order = ['feature_set', 'model', 'r2', 'mae', 'rmse', 'best_params']
            sort_by_col = 'r2'

        # Use only the columns that actually exist in the current results DataFrame
        results_df = results_df[[col for col in column_order if col in results_df.columns]]
        # Sort the results
        results_df = results_df.sort_values(by=['feature_set', sort_by_col], ascending=[True, False])

        try:
            # Save in the 'postprandial_db' folder
            db_directory = os.path.dirname(self.data_file_path) # Get the directory first
            path = os.path.join(db_directory, filename) # Build path from directory
            results_df.to_csv(path, index=False)
            print(f"\nSuccessfully saved model performance summary to: {path}")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")

      

    def train(self):
        """
        Main orchestration method for the training pipeline.
        """
        if self.df.empty or not self.feature_sets:
            print("Cannot start training. Data or feature sets are missing.")
            return

        print("\n--- Starting Model Training Pipeline ---")
        
        # Step 1: Split the data
        self._split_data_by_subject()
        
        if self.train_df is None or self.test_df is None:
            print("Data splitting failed. Aborting training.")
            return
        
        models_to_train = self._get_models()
        param_grids = self._get_parameter_grids()

        # Step 2: Loop through each pre-selected feature set and train models
        for feature_set_name, feature_list in self.feature_sets.items():
            print(f"\n{'='*20} Feature Set: '{feature_set_name}' {'='*20}")

            if not feature_list:
                print("Feature list is empty, skipping.")
                continue
            
            X_train_initial = self.train_df[feature_list]
            X_test_initial = self.test_df[feature_list]
            
            if self.analysis_type == 'binary_classification':
                print("Preparing target for BINARY classification.")
                y_train_initial = self.train_df[self.classification_target].apply(lambda x: 1 if x > 0 else 0)
                y_test_initial = self.test_df[self.classification_target].apply(lambda x: 1 if x > 0 else 0)
            elif self.analysis_type == 'multi_class_classification':
                print("Preparing target for MULTI-CLASS classification.")
                y_train_initial = self.train_df[self.classification_target]
                y_test_initial = self.test_df[self.classification_target]
            else: # Regression
                y_train_initial = self.train_df[self.regression_target]
                y_test_initial = self.test_df[self.regression_target]

            # Drop rows where the target is NaN from ALL datasets (X, y, and groups)
            train_full = pd.concat([X_train_initial, y_train_initial, self.train_df['subj_id']], axis=1).dropna(subset=[y_train_initial.name])
            X_train = train_full[feature_list]
            y_train = train_full[y_train_initial.name]
            groups = train_full['subj_id'] # CORRECT groups are defined here

            # Also filter the test set for consistency in evaluation
            test_full = pd.concat([X_test_initial, y_test_initial], axis=1).dropna(subset=[y_test_initial.name])
            X_test = test_full[feature_list]
            y_test = test_full[y_test_initial.name]

            # Block to automatically update the XGBoost grid
            if self.analysis_type == 'binary_classification' and "XGBoost" in param_grids:
                imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
                print(f"Calculated Imbalance Ratio (for XGBoost scale_pos_weight): {imbalance_ratio:.2f}")
                # Add the ideal ratio to the list of values to test
                if 'model__scale_pos_weight' in param_grids["XGBoost"]:
                    param_grids["XGBoost"]['model__scale_pos_weight'].append(imbalance_ratio)
                

            for model_name, model in models_to_train.items():
                print(f"\n>>> Training Model: {model_name}...")

                # Create a pipeline to handle missing values, scale data, and train the model
                # Use pipline so that one object handles everything
                pipeline = Pipeline([ # Pipeline to chain multiple preprocessing steps and a model them into one unified workflow
                    ('imputer', SimpleImputer(strategy='median')), # Fills missing value ---> SimpleImputer(strategy='median') replaces missing values in each feature column with the median of that column
                    ('scaler', StandardScaler()),                  # Scales features
                    ('model', model)                               # The actual model ---> training and prediction automatically include preprocessing
                ])
                
                param_grid = param_grids.get(model_name)
                if not param_grid:
                    print(f"No parameter grid found for {model_name}. Skipping tuning.")
                    continue

                # TUNING AND TRAINING
                cv = GroupKFold(n_splits=self.n_splits) # no subject used for the training are in the test set ---> no data leakage
                
                # 1. Tune the model to find the best hyperparameters
                best_pipeline, tuning_results = self._perform_grid_search(pipeline, param_grid, X_train, y_train, cv, groups)
                
                # Save the detailed tuning results for this model
                tuning_filename = f"tuning_results_{model_name.replace(' ', '')}_{feature_set_name}.csv"
                try:
                    db_directory = os.path.dirname(self.data_file_path) 
                    path = os.path.join(db_directory, 'tuning_logs', tuning_filename) 
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tuning_results.to_csv(path, index=False)
                    print(f"Saved detailed tuning results to: {path}")
                except Exception as e:
                    print(f"Error saving tuning results: {e}")
                
                # 2. Use the best found model for final evaluation on the test set
                y_pred = best_pipeline.predict(X_test)
                
                # Call the correct evaluation function 
                if 'classification' in self.analysis_type:
                    y_pred_proba = best_pipeline.predict_proba(X_test) 
                    results = self._evaluate_classification(model_name, best_pipeline.get_params()['model'], y_test, y_pred, y_pred_proba)
                else: # Regression
                    results = self._evaluate_regression(model_name, best_pipeline.get_params()['model'], y_test, y_pred)
                     
                # 3. Log the results of the tested models
                results['feature_set'] = feature_set_name # Add feature set info
                self.results_log.append(results)

        print("\nModel training pipeline finished.")









if __name__ == '__main__':
    analysis_mode = 'classification' 
    
    classification_type = 'multi_class' # 'binary' or 'multi_class'
    trainer_analysis_type = f'{classification_type}_classification'
    
    features_filename = f"{classification_type}_classification_selected_feature_sets.json"

    trainer = ModelTrainer(
        data_file_path='postprandial_db/all_postprandial_data_with_features.csv',
        features_file_path=f'postprandial_db/{features_filename}',
        analysis_type=trainer_analysis_type, 
        classification_target='target_hypoglycemia',
        regression_target='postprandial_glucose_min'
    )
    trainer.train()