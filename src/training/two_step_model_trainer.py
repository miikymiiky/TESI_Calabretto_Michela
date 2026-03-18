#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 23:28:19 2025

@author: michela
"""

import pandas as pd
import numpy as np
import os
import json
import random
from sklearn.model_selection import GroupKFold, GridSearchCV, ParameterGrid
#from imblearn.over_sampling import SMOTE                   
#from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Model imports
#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression



class TwoStepClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom two-step classifier that is compatible with scikit-learn's GridSearchCV.
    1. A binary classifier to predict 0 vs {1, 2}.
    2. A binary classifier to distinguish between 1 and 2.
    """
    
    _estimator_type = "classifier" 
    
    def __init__(self, classifier1, classifier2):
        self.classifier1 = classifier1
        self.classifier2 = classifier2
        
        # Check if classifier2 has been fitted by looking for a standard attribute.
        # A fitted scikit-learn classifier will have the 'classes_' attribute.
        # This ensures the 'is_step2_trained_' flag is set upon creation.
        if hasattr(self.classifier2, 'classes_'):
            self.is_step2_trained_ = True
        else:
            self.is_step2_trained_ = False
        
        # Define the mapping dictionaries upon initialization.
        # They are constant and do not depend on the training data.
        self.y2_map = {1: 0, 2: 1}
        self.y2_inv_map = {0: 1, 1: 2}



    def fit(self, X, y):
        # Set self.classes_ UNCONDITIONALLY at the start of fit.
        self.classes_ = np.unique(y)

        # Step 1: Train the first classifier (0 vs. non-zero)
        y1 = (y > 0).astype(int)
        self.classifier1.fit(X, y1)

        # Step 2: Train the second classifier only on data for classes 1 and 2
        mask_hypo = (y > 0)
        X2 = X[mask_hypo]
        y2_original = y[mask_hypo]

        if len(X2) > 0 and y2_original.nunique() > 1:
            # Remap labels {1, 2} to {0, 1} for the binary classifier
            self.y2_map = {1: 0, 2: 1}
            self.y2_inv_map = {0: 1, 1: 2}
            y2_mapped = y2_original.map(self.y2_map)
            
            self.classifier2.fit(X2, y2_mapped)
            self.is_step2_trained_ = True
        else:
            self.is_step2_trained_ = False
            print("Warning: Not enough data or only one class present for step-2 classifier. It will not be trained.")
        
        return self



    def predict(self, X):
        preds1 = self.classifier1.predict(X)
        final_preds = np.zeros_like(preds1) # creates an array of 0s, where the predicted labels will replace the placeholder 0sp
        
        hypo_indices = np.where(preds1 == 1)[0]
        
        if len(hypo_indices) > 0 and self.is_step2_trained_:
            X_step2 = X.iloc[hypo_indices] if isinstance(X, pd.DataFrame) else X[hypo_indices]
            preds2_mapped = self.classifier2.predict(X_step2)
            # Convert predictions back to original labels {1, 2}
            preds2_original = np.array([self.y2_inv_map[p] for p in preds2_mapped])
            final_preds[hypo_indices] = preds2_original
        elif len(hypo_indices) > 0:
            # If step 2 model isn't trained, default to predicting class 1
            final_preds[hypo_indices] = 1
            
        return final_preds



    def predict_proba(self, X):
        proba1 = self.classifier1.predict_proba(X)
        final_proba = np.zeros((X.shape[0], 3))

        # P(class=0) comes directly from the first model
        # Assigns the No Hypo probability directly from classifier1 to the first column of final_proba.
        final_proba[:, 0] = proba1[:, 0]
        
        if self.is_step2_trained_:
            # This is slower as it predicts for all X, but necessary for probability calculation
            proba2 = self.classifier2.predict_proba(X)
            
            # P(class=1) = P(is_hypo) * P(is_class_1 | is_hypo)
            final_proba[:, 1] = proba1[:, 1] * proba2[:, 0] # proba2[:, 0] is prob of mapped class 0 (original 1)
            # P(class=2) = P(is_hypo) * P(is_class_2 | is_hypo)
            final_proba[:, 2] = proba1[:, 1] * proba2[:, 1] # proba2[:, 1] is prob of mapped class 1 (original 2)
        else:
            final_proba[:, 1] = proba1[:, 1] # All hypo probability goes to class 1
            final_proba[:, 2] = 0

        return final_proba
    
    
    
    def get_params(self, deep=True):
        return super().get_params(deep=deep)



    def set_params(self, **params):
        return super().set_params(**params)
    







class TwoStepModelTrainer:
    """
    Trains and tunes various two-step classification models using GridSearchCV.
    Model combinations are defined internally.
    """
    def __init__(self, data_file_path: str, features_file_path: str,
                 classification_target: str, test_size: float = 0.2, 
                 random_state: int = 42, n_splits: int = 5):
        
        self.data_file_path = data_file_path
        self.features_file_path = features_file_path
        self.classification_target = classification_target
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = n_splits
        
        self.df = self._load_data()
        self.df = self._apply_pre_split_filters(self.df) # Apply filters before splitting
        self.feature_sets = self._load_feature_sets()
        self._display_loaded_feature_sets()
        self.results_log = []
        self.best_model = None
    
    
    
    
    def _display_loaded_feature_sets(self):
        """
        Prints the loaded feature sets in a clear, formatted way for verification.
        """
        if not self.feature_sets:
            print("\nWARNING: No feature sets were loaded for this training run.")
            return

        print("\n" + "#"*80)
        print("### FEATURE SETS LOADED FOR THIS TRAINING RUN ###")
        print(f"Source File: {self.features_file_path}")
        print("#"*80)

        for set_name, features in self.feature_sets.items():
            print("\n" + "="*80)
            print(f"--- Feature Set: '{set_name}' ---")
            print(f"Total Features: {len(features)}")
            print("="*80)
            if not features:
                print("  (No features in this set)")
            else:
                for i, feature in enumerate(features, 1):
                    print(f"  {i:2d}. {feature}")
        print("\n" + "#"*80 + "\n")
    
    
    
    def _apply_pre_split_filters(self, df):
        """Applies the standard project filters to the dataframe before data splitting."""
        if df.empty:
            return df

        print("\n--- Applying Pre-Split Filters to the Dataset ---")
        initial_rows = len(df)

        # These filters are copied from main.py to ensure consistency
        non_wearable_filters = [
            {'column': 'hypo_to_check_flag', 'operator': '!=', 'value': 0},
            {'column': 'postprandial_duration_minutes', 'operator': '>', 'value': 120},
            {'column': 'time_peak_hypo_event', 'operator': '>', 'value': -1},
            {'column': 'rate_of_increase', 'operator': '>', 'value': 0},
            {'column': 'glucose_derivative_15min_before_peak1', 'operator': '>', 'value': 0},
            {'column': 'glucose_derivative_15min_after_peak1', 'operator': '<', 'value': 0},
        ]
        wearable_filters = [
            {'column': 'has_heart_rate_data', 'operator': '==', 'value': 1},
            {'column': 'has_pulse_ox_data', 'operator': '==', 'value': 1},
            {'column': 'has_steps_data', 'operator': '==', 'value': 1},
        ]
        all_filters = non_wearable_filters + wearable_filters

        # Apply each filter condition
        for condition in all_filters:
            col = condition['column']
            op = condition['operator']
            val = condition['value']
            
            if col not in df.columns:
                print(f"Warning: Filter column '{col}' not found. Skipping filter.")
                continue

            if op == '>': df = df[df[col] > val]
            elif op == '<': df = df[df[col] < val]
            elif op == '==': df = df[df[col] == val]
            elif op == '>=': df = df[df[col] >= val]
            elif op == '<=': df = df[df[col] <= val]
            elif op == '!=': df = df[df[col] != val]

        final_rows = len(df)
        print(f"Filtered dataset from {initial_rows} to {final_rows} rows before splitting.")
        
        print("\n--- Target Class Distribution After Filtering ---")
        if self.classification_target in df.columns:
            print(df[self.classification_target].value_counts())
            
        return df



    def _get_model_combinations(self) -> dict:
        """
        Defines the dictionary of model combinations and parameter grids to be tested.
        """
        return {
            "XGB_to_XGB": {
                "classifier1": XGBClassifier(random_state=self.random_state, eval_metric='logloss', scale_pos_weight=5.0),
                "params1": {
                    'n_estimators': [50, 100, 200], 
                    'max_depth': [1, 2, 3, 4, 5], 
                    'scale_pos_weight': [1, 4, 5]
                },
                "classifier2": XGBClassifier(random_state=self.random_state, eval_metric='logloss', scale_pos_weight=5.0),
                "params2": {
                    'n_estimators': [30, 50, 100], 
                    'max_depth': [1, 2, 3]
                }
            },
            "XGB_to_LogReg": {
                "classifier1": XGBClassifier(random_state=self.random_state, eval_metric='logloss', scale_pos_weight=5.0),
                "params1": {
                    'n_estimators': [50, 100, 200], 
                    'max_depth': [1, 2, 3, 4, 5], 
                    'scale_pos_weight': [4, 5]
                },
                "classifier2": LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
                "params2": {
                    'penalty': ['l1', 'l2'], 
                    'C': [0.1, 1, 10], 
                    'solver': ['liblinear']
                }
            },
            "LogReg_to_XGB": {
                "classifier1": LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
                "params1": {
                    'penalty': ['l1', 'l2'],
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear', 'saga']
                },
                "classifier2": XGBClassifier(random_state=self.random_state, eval_metric='logloss', scale_pos_weight=5.0),
                "params2": {
                    'n_estimators': [30, 50, 100],
                    'max_depth': [1, 2, 3]
                }
            },
            "LogReg_to_LogReg": {
                "classifier1": LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
                "params1": {
                    'penalty': ['l2'],
                    'C': [0.1, 1, 10],
                    'solver': ['saga']
                },
                "classifier2": LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
                "params2": {
                    'penalty': ['l1'],
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['liblinear']
                }
            }
        }



    def _load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_file_path)
            print(f"Successfully loaded data from {self.data_file_path}")
            return df.dropna(subset=[self.classification_target])
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()



    def _load_feature_sets(self) -> dict:
        try:
            with open(self.features_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR: Could not load features file: {e}")
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
        
    
    
    # --- CHANGES START ---
    
    def _tune_classifier1(self, X_train, y_train, groups, pipeline, param_grid):
        print("--- Fitting Stage 1 Classifier (Skipping Tuning) ---")
        y1_train = (y_train > 0).astype(int) # Target for stage 1
        pipeline.fit(X_train, y1_train)
        return pipeline

    def _tune_classifier2(self, X_train, y_train, groups, pipeline, param_grid):
        print("--- Fitting Stage 2 Classifier (Skipping Tuning) ---")
        
        # Prepare data for stage 2 (only hypo events)
        mask_hypo = y_train > 0
        X2_train = X_train[mask_hypo]
        y2_train_original = y_train[mask_hypo]
        y2_train_mapped = y2_train_original.map({1: 0, 2: 1})

        if y2_train_mapped.nunique() < 2:
            print("Warning: Only one class present for Stage 2. Skipping fitting.")
            # Return a dummy or unfitted pipeline
            return pipeline 
        
        pipeline.fit(X2_train, y2_train_mapped)
        return pipeline
    
    
    
    # TUNING & CV CODE TO COMMENT OUT FOR NOW
    
    # def _tune_classifier1(self, X_train, y_train, groups, pipeline, param_grid):
    #     print("--- Tuning Stage 1 Classifier (Optimizing for Recall) ---")
    #     cv = GroupKFold(n_splits=self.n_splits) # ensures that during CV the same group is not represented in both training and testing folds
    #     param_iterator = ParameterGrid(param_grid) # ParameterGrid(param_grid) ---> expands a dictionary of parameters into all possible combinations
    #     best_score = -1
    #     best_params = None

    #     y1_train = (y_train > 0).astype(int) # Target for stage 1
        
    #     # Loop that iterates through each parameter set
    #     for params in param_iterator:
    #         fold_scores = []
    #         for train_idx, test_idx in cv.split(X_train, y1_train, groups): # cv.split() ---> splits the data into train/test indices according to groups
    #             # Create a fresh copy of the pipeline so that previous training doesn’t carry over
    #             cloned_pipeline = clone(pipeline)
    #             cloned_pipeline.set_params(**params) # inject the current parameter set into the pipeline
                
    #             # Manually impute and scale data for this fold
    #             # Use the fold indices to create training and test subsets for that fold (split into training and test set)
    #             imputer = SimpleImputer(strategy='median'); scaler = StandardScaler()
    #             X_train_fold = scaler.fit_transform(imputer.fit_transform(X_train.iloc[train_idx])); y_train_fold = y1_train.iloc[train_idx]
    #             X_test_fold = scaler.transform(imputer.transform(X_train.iloc[test_idx])); y_test_fold = y1_train.iloc[test_idx]
               
    #             # # Use the fold indices to create training and test subsets for that fold (split into training and test set)
    #             # X_train_fold, y_train_fold = X_train.iloc[train_idx], y1_train.iloc[train_idx]
    #             # X_test_fold, y_test_fold = X_train.iloc[test_idx], y1_train.iloc[test_idx]

    #             cloned_pipeline.fit(X_train_fold, y_train_fold)
    #             y_pred = cloned_pipeline.predict(X_test_fold)
                
    #             # The parameter set with the best recall score will be the chosen one for the first classifier
    #             # Recall for the positive class (class 1: any hypo)
    #             score = recall_score(y_test_fold, y_pred, pos_label=1, zero_division=0)
    #             fold_scores.append(score)
            
    #         mean_score = np.mean(fold_scores) # append NaN when it fails
    #         if mean_score > best_score:
    #             best_score = mean_score
    #             best_params = params

    #     print(f"Best Recall for Stage 1: {best_score:.4f}")
    #     print(f"Best Params for Stage 1: {best_params}")

    #     best_pipeline_c1 = clone(pipeline).set_params(**best_params)
    #     best_pipeline_c1.fit(X_train, y1_train) # refit on all training data
    #     return best_pipeline_c1

    

    # def _tune_classifier2(self, X_train, y_train, groups, pipeline, param_grid):
    #     print("--- Tuning Stage 2 Classifier (Optimizing for AUC) ---")
    #     cv = GroupKFold(n_splits=self.n_splits) # ensures that during CV the same group is not represented in both training and testing folds
    #     param_iterator = ParameterGrid(param_grid) # ParameterGrid(param_grid) ---> expands a dictionary of parameters into all possible combinations
    #     best_score = -1
    #     best_params = None

    #     # Prepare data for stage 2 (only hypo events)
    #     mask_hypo = y_train > 0
    #     X2_train = X_train[mask_hypo]
    #     y2_train_original = y_train[mask_hypo]
    #     y2_train_mapped = y2_train_original.map({1: 0, 2: 1})
    #     groups2 = groups[mask_hypo]

    #     if len(y2_train_mapped.unique()) < 2:
    #         print("Warning: Only one class present for Stage 2 tuning. Skipping.")
    #         return clone(pipeline).fit(X2_train, y2_train_mapped) # Return a default fitted model
        
    #     # Loop that iterates through each parameter set
    #     for params in param_iterator:
    #         fold_scores = []
    #         for train_idx, test_idx in cv.split(X2_train, y2_train_mapped, groups2): # cv.split() ---> splits the data into train/test indices according to groups
    #             # Create a fresh copy of the pipeline so that previous training doesn’t carry over
    #             cloned_pipeline = clone(pipeline)
    #             cloned_pipeline.set_params(**params) # inject the current parameter set into the pipeline
                
    #             # Manually impute and scale data for this fold
    #             # Use the fold indices to create training and test subsets for that fold (split into training and test set)
    #             imputer = SimpleImputer(strategy='median'); scaler = StandardScaler()
    #             X_train_fold = scaler.fit_transform(imputer.fit_transform(X2_train.iloc[train_idx])); y_train_fold = y2_train_mapped.iloc[train_idx]
    #             X_test_fold = scaler.transform(imputer.transform(X2_train.iloc[test_idx])); y_test_fold = y2_train_mapped.iloc[test_idx]
              
    #             # # Use the fold indices to create training and test subsets for that fold (split into training and test set)
    #             # X_train_fold, y_train_fold = X2_train.iloc[train_idx], y2_train_mapped.iloc[train_idx]
    #             # X_test_fold, y_test_fold = X2_train.iloc[test_idx], y2_train_mapped.iloc[test_idx]
                
    #             cloned_pipeline.fit(X_train_fold, y_train_fold)
                
    #             # Check if both classes are in the test fold before scoring AUC
    #             if len(np.unique(y_test_fold)) > 1:
    #                 y_pred_proba = cloned_pipeline.predict_proba(X_test_fold)[:, 1]
    #                 # The parameter set with the best AUC score will be the chosen one for the second classifier
    #                 score = roc_auc_score(y_test_fold, y_pred_proba)
    #                 fold_scores.append(score)
            
    #         if fold_scores: # Only if we could calculate scores
    #             mean_score = np.mean(fold_scores) # append NaN when it fails
    #             if mean_score > best_score:
    #                 best_score = mean_score
    #                 best_params = params
        
    #     print(f"Best AUC for Stage 2: {best_score:.4f}")
    #     print(f"Best Params for Stage 2: {best_params}")

    #     best_pipeline_c2 = clone(pipeline).set_params(**best_params)
    #     best_pipeline_c2.fit(X2_train, y2_train_mapped) # refit on all training data
    #     return best_pipeline_c2

    # --- CHANGES FINISH ---
    


    def evaluate(self, X_test, y_true, test_df_full, model_combo_name: str, feature_set_name: str):
        print(f"\n--- Evaluating on Test Set ({model_combo_name}) ---")
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=[0, 1, 2], target_names=['No Hypo', 'Hypo', 'Deep Hypo'])
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')

        print("\n--- Multi-Class Classification Performance ---")
        print(f"Weighted OvR AUC-ROC: {auc_score:.4f}")
        print(classification_report(y_true, y_pred, target_names=['No Hypo', 'Hypo', 'Deep Hypo'], zero_division=0))
        
        # Time Gain Calculation
        true_positives_mask = (y_true.values > 0) & (y_pred > 0)
        mean_time_gain, median_time_gain = np.nan, np.nan
        print("\n--- Time Gain Analysis (for True Positives) ---")
        if np.any(true_positives_mask):
            tp_df = test_df_full.loc[true_positives_mask].copy()
            
            # Merge the hypo time back in using the original index
            hypo_time_series = self.df['postprandial_hypo_time']
            tp_df = tp_df.merge(hypo_time_series, left_index=True, right_index=True, how='left')

            tp_df['meal_time_start'] = pd.to_datetime(tp_df['meal_time_start'])
            tp_df['postprandial_hypo_time'] = pd.to_datetime(tp_df['postprandial_hypo_time'])

            prediction_time = tp_df['meal_time_start'] + pd.to_timedelta(tp_df['time_to_first_glucose_peak_after_meal'] + 15, unit='m')
            time_gain = (tp_df['postprandial_hypo_time'] - prediction_time).dt.total_seconds() / 60
            time_gain = time_gain[time_gain > 0]
            
            if not time_gain.empty:
                mean_time_gain = time_gain.mean()
                median_time_gain = time_gain.median()
                print(f"Number of useful predictions (TPs made before event): {len(time_gain)} / {true_positives_mask.sum()}")
                print(f"Average Time Gain: {mean_time_gain:.1f} minutes")
                print(f"Median Time Gain:  {median_time_gain:.1f} minutes")
            else:
                print("No true positive predictions were made before the event occurred.")
        else:
            print("No True Positives to analyze.")
        
        self.results_log.append({
            'model_combination': model_combo_name,
            'feature_set': feature_set_name,
            'auc_roc_ovr_weighted': auc_score,
            'recall_hypo': report_dict['Hypo']['recall'],
            'recall_deep_hypo': report_dict['Deep Hypo']['recall'],
            'precision_hypo': report_dict['Hypo']['precision'],
            'precision_deep_hypo': report_dict['Deep Hypo']['precision'],
            'f1_weighted': report_dict['weighted avg']['f1-score'],
            'mean_time_gain_min': mean_time_gain,
            'median_time_gain_min': median_time_gain
        })



    def save_results_to_csv(self, filename="two_step_performance_log.csv"):
        if not self.results_log:
            print("No results to save.")
            return
        results_df = pd.DataFrame(self.results_log)
        db_directory = os.path.dirname(self.data_file_path)
        path = os.path.join(db_directory, filename)
        results_df.to_csv(path, index=False)
        print(f"\nSuccessfully saved performance summary to: {path}")



    def run(self, use_smote=False):
        if self.df.empty or not self.feature_sets:
            print("Cannot run pipeline. Data or feature sets are missing.")
            return

        self._split_data_by_subject()
        model_combinations = self._get_model_combinations()
        
        for feature_set_name, feature_list in self.feature_sets.items():
            print(f"\n{'='*25} Processing Feature Set: '{feature_set_name}' {'='*25}")
            
            if not all(feat in self.df.columns for feat in feature_list):
                print(f"Warning: Skipping feature set '{feature_set_name}' due to missing columns.")
                continue
            

            # --- Handle NaNs for both train and test sets ---
            # Training set
            X_train_initial = self.train_df[feature_list]
            y_train_initial = self.train_df[self.classification_target].astype(int)
            # Add the required time columns for the Time Gain calculation
            train_full = pd.concat([X_train_initial, y_train_initial, self.train_df['subj_id'], self.train_df['meal_time_start'], self.train_df['time_to_first_glucose_peak_after_meal']], axis=1)
            initial_train_rows = len(train_full)
            train_full.dropna(inplace=True)
            
            print(f"Dropped {initial_train_rows - len(train_full)} rows with NaN values from the training set.")
            print(f"Final training set dimensions: {train_full.shape[0]} postprandial periods.")
            print("Final class distribution in training set:")
            print(train_full[self.classification_target].value_counts())

            # Test set
            X_test_initial = self.test_df[feature_list]
            y_test_initial = self.test_df[self.classification_target].astype(int)
            # Add the required time columns for the Time Gain calculation
            test_full = pd.concat([X_test_initial, y_test_initial, self.test_df['subj_id'], self.test_df['meal_time_start'], self.test_df['time_to_first_glucose_peak_after_meal']], axis=1)
            initial_test_rows = len(test_full)
            test_full.dropna(inplace=True)
            
            print(f"\nDropped {initial_test_rows - len(test_full)} rows with NaN values from the test set.")
            print(f"Final test set dimensions: {test_full.shape[0]} postprandial periods.")
            print("Final class distribution in test set:")
            print(test_full[self.classification_target].value_counts())

            # Calculate and print final split percentage
            final_train_rows = len(train_full)
            final_test_rows = len(test_full)
            total_rows = final_train_rows + final_test_rows
            if total_rows > 0:
                train_percent = (final_train_rows / total_rows) * 100
                test_percent = (final_test_rows / total_rows) * 100
                print(f"\nFinal data split after dropping NaNs: {train_percent:.1f}% train / {test_percent:.1f}% test")
            
            if final_train_rows == 0 or final_test_rows == 0:
                print("Not enough data to train/test after dropping NaNs. Skipping feature set.")
                continue

            # Prepare final, clean data for the methods
            X_train = train_full[feature_list]
            y_train = train_full[self.classification_target]
            groups = train_full['subj_id']
            X_test = test_full[feature_list]
            y_test = test_full[self.classification_target]


            for model_combo_name, models in model_combinations.items():
                print(f"\n--- Testing Model Combination: {model_combo_name} ---")
                
                # Create separate pipeline steps for each stage
                pipeline_steps_c1 = [
                    #('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]
               
                pipeline_steps_c2 = [
                    #('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]

                if use_smote:
                    print(">>> SMOTE enabled for both classifier stages.")
                    pipeline_steps_c1.insert(1, ('sampler', SMOTE(random_state=self.random_state)))
                    pipeline_steps_c2.insert(1, ('sampler', SMOTE(random_state=self.random_state)))

                pipeline_steps_c1.append(('model', models['classifier1']))
                pipeline_steps_c2.append(('model', models['classifier2']))

                pipeline_c1 = ImbPipeline(pipeline_steps_c1)
                pipeline_c2 = ImbPipeline(pipeline_steps_c2)

                # Create parameter grids with the 'model__' prefix for the pipeline
                params1_grid = {'model__' + k: v for k, v in models['params1'].items()} 
                params2_grid = {'model__' + k: v for k, v in models['params2'].items()}

                # TUNING 1: Get the FITTED pipeline for the first classifier
                best_c1_pipeline = self._tune_classifier1(X_train, y_train, groups, pipeline_c1, params1_grid)

                # TUNING 2: Get the FITTED pipeline for the second classifier
                best_c2_pipeline = self._tune_classifier2(X_train, y_train, groups, pipeline_c2, params2_grid)

                # Assemble the final model by extracting the ALREADY FITTED estimators
                best_c1 = best_c1_pipeline.named_steps['model']
                best_c2 = best_c2_pipeline.named_steps['model']

                # Create the final two-step model object. This object does NOT need to be fitted again.
                # We just need it to provide a unified predict/predict_proba interface.
                # The scaler is taken from the first pipeline for consistency.
                self.best_model = ImbPipeline([
                    ('scaler', best_c1_pipeline.named_steps['scaler']),
                    ('model', TwoStepClassifier(classifier1=best_c1, classifier2=best_c2))
                ])
   
                # Evaluate the final assembled model
                self.evaluate(X_test, y_test, test_full, model_combo_name, feature_set_name)
        
        # Conditionally name the log file and save the results
        log_filename = "tuned_two_step_performance_log.csv"
        if use_smote:
            log_filename = "tuned_two_step_performance_log_with_smote.csv"
        
        self.save_results_to_csv(log_filename)

 
                
                
                
                
                
                