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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Model imports
from xgboost import XGBClassifier
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
        self.feature_sets = self._load_feature_sets()
        self.results_log = []
        self.best_model = None



    def _get_model_combinations(self) -> dict:
        """
        Defines the dictionary of model combinations and parameter grids to be tested.
        """
        return {
            "XGB_to_XGB": {
                "classifier1": XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                "params1": {
                    'n_estimators': [50, 100, 200], 
                    'max_depth': [1, 2, 3, 4, 5], 
                    'scale_pos_weight': [1, 4, 5]
                },
                "classifier2": XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                "params2": {
                    'n_estimators': [30, 50, 100], 
                    'max_depth': [1, 2, 3]
                }
            },
            "XGB_to_LogReg": {
                "classifier1": XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
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
                "classifier2": XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
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
    
    
    
    # def _get_model_combinations(self) -> dict:
    #     """
    #     Defines the dictionary of model combinations and parameter grids to be tested.
    #     """
    #     return {
    #         "XGB_to_XGB": {
    #             "model": TwoStepClassifier(
    #                 classifier1=XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
    #                 classifier2=XGBClassifier(random_state=self.random_state, eval_metric='logloss')
    #             ),
    #             "params": {
    #                 'model__classifier1__n_estimators': [50, 100, 200], 
    #                 'model__classifier1__max_depth': [1, 2, 3, 4, 5], 
    #                 'model__classifier1__scale_pos_weight': [1, 4, 5],
                    
    #                 'model__classifier2__n_estimators': [30, 50, 100], 
    #                 'model__classifier2__max_depth': [1, 2, 3]
    #             }
    #         },
    #         "XGB_to_LogReg": {
    #             "model": TwoStepClassifier(
    #                 classifier1=XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
    #                 classifier2=LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced')
    #             ),
    #             "params": {
    #                 'model__classifier1__n_estimators': [50, 100, 200], 
    #                 'model__classifier1__max_depth': [1, 2, 3, 4, 5], 
    #                 'model__classifier1__scale_pos_weight': [4, 5],
                    
    #                 'model__classifier2__penalty': ['l1', 'l2'], 
    #                 'model__classifier2__C': [0.01, 0.1, 1, 10], 
    #                 'model__classifier2__solver': ['liblinear']
    #             }
    #         },
    #         "LogReg_to_XGB": {
    #             "model": TwoStepClassifier(
    #                 classifier1=LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
    #                 classifier2=XGBClassifier(random_state=self.random_state, eval_metric='logloss')
    #             ),
    #             "params": [
    #                 {
    #                     'model__classifier1__penalty': ['l1', 'l2'],
    #                     'model__classifier1__C': [0.01, 0.1, 1, 10],
    #                     'model__classifier1__solver': ['liblinear'],
                        
    #                     'model__classifier2__n_estimators': [30, 50, 100],
    #                     'model__classifier2__max_depth': [1, 2, 3]
    #                 },
    #                 {
    #                     'model__classifier1__penalty': ['l1', 'l2'],
    #                     'model__classifier1__C': [0.01, 0.1, 1, 10],
    #                     'model__classifier1__solver': ['saga'],
                        
    #                     'model__classifier2__n_estimators': [30, 50, 100],
    #                     'model__classifier2__max_depth': [1, 2, 3]
    #                 }
    #             ]
    #         },
    #         "LogReg_to_LogReg": {
    #             "model": TwoStepClassifier(
    #                 classifier1=LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
    #                 classifier2=LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced')
    #             ),
    #             "params": {
    #                 'model__classifier1__penalty': ['l2'],
    #                 'model__classifier1__C': [0.01, 0.1, 1, 10],
    #                 'model__classifier1__solver': ['saga'],
                    
    #                 'model__classifier2__penalty': ['l1'],
    #                 'model__classifier2__C': [0.01, 0.1, 1, 10],
    #                 'model__classifier2__solver': ['liblinear']
    #             }
    #         }
    #     }



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
        
    
    
    def _tune_classifier1(self, X_train, y_train, groups, pipeline, param_grid):
        print("--- Tuning Stage 1 Classifier (Optimizing for Recall) ---")
        cv = GroupKFold(n_splits=self.n_splits) # ensures that during CV the same group is not represented in both training and testing folds
        param_iterator = ParameterGrid(param_grid) # ParameterGrid(param_grid) ---> expands a dictionary of parameters into all possible combinations
        best_score = -1
        best_params = None

        y1_train = (y_train > 0).astype(int) # Target for stage 1
        
        # Loop that iterates through each parameter set
        for params in param_iterator:
            fold_scores = []
            for train_idx, test_idx in cv.split(X_train, y1_train, groups): # cv.split() ---> splits the data into train/test indices according to groups
                # Create a fresh copy of the pipeline so that previous training doesn’t carry over
                cloned_pipeline = clone(pipeline)
                cloned_pipeline.set_params(**params) # inject the current parameter set into the pipeline
                
                # Manually impute and scale data for this fold
                # Use the fold indices to create training and test subsets for that fold (split into training and test set)
                imputer = SimpleImputer(strategy='median'); scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(imputer.fit_transform(X_train.iloc[train_idx])); y_train_fold = y1_train.iloc[train_idx]
                X_test_fold = scaler.transform(imputer.transform(X_train.iloc[test_idx])); y_test_fold = y1_train.iloc[test_idx]
               
                # # Use the fold indices to create training and test subsets for that fold (split into training and test set)
                # X_train_fold, y_train_fold = X_train.iloc[train_idx], y1_train.iloc[train_idx]
                # X_test_fold, y_test_fold = X_train.iloc[test_idx], y1_train.iloc[test_idx]

                cloned_pipeline.fit(X_train_fold, y_train_fold)
                y_pred = cloned_pipeline.predict(X_test_fold)
                
                # The parameter set with the best recall score will be the chosen one for the first classifier
                # Recall for the positive class (class 1: any hypo)
                score = recall_score(y_test_fold, y_pred, pos_label=1, zero_division=0)
                fold_scores.append(score)
            
            mean_score = np.mean(fold_scores) # append NaN when it fails
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        print(f"Best Recall for Stage 1: {best_score:.4f}")
        print(f"Best Params for Stage 1: {best_params}")

        best_pipeline_c1 = clone(pipeline).set_params(**best_params)
        best_pipeline_c1.fit(X_train, y1_train) # refit on all training data
        return best_pipeline_c1

    

    def _tune_classifier2(self, X_train, y_train, groups, pipeline, param_grid):
        print("--- Tuning Stage 2 Classifier (Optimizing for AUC) ---")
        cv = GroupKFold(n_splits=self.n_splits) # ensures that during CV the same group is not represented in both training and testing folds
        param_iterator = ParameterGrid(param_grid) # ParameterGrid(param_grid) ---> expands a dictionary of parameters into all possible combinations
        best_score = -1
        best_params = None

        # Prepare data for stage 2 (only hypo events)
        mask_hypo = y_train > 0
        X2_train = X_train[mask_hypo]
        y2_train_original = y_train[mask_hypo]
        y2_train_mapped = y2_train_original.map({1: 0, 2: 1})
        groups2 = groups[mask_hypo]

        if len(y2_train_mapped.unique()) < 2:
            print("Warning: Only one class present for Stage 2 tuning. Skipping.")
            return clone(pipeline).fit(X2_train, y2_train_mapped) # Return a default fitted model
        
        # Loop that iterates through each parameter set
        for params in param_iterator:
            fold_scores = []
            for train_idx, test_idx in cv.split(X2_train, y2_train_mapped, groups2): # cv.split() ---> splits the data into train/test indices according to groups
                # Create a fresh copy of the pipeline so that previous training doesn’t carry over
                cloned_pipeline = clone(pipeline)
                cloned_pipeline.set_params(**params) # inject the current parameter set into the pipeline
                
                # Manually impute and scale data for this fold
                # Use the fold indices to create training and test subsets for that fold (split into training and test set)
                imputer = SimpleImputer(strategy='median'); scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(imputer.fit_transform(X2_train.iloc[train_idx])); y_train_fold = y2_train_mapped.iloc[train_idx]
                X_test_fold = scaler.transform(imputer.transform(X2_train.iloc[test_idx])); y_test_fold = y2_train_mapped.iloc[test_idx]
              
                # # Use the fold indices to create training and test subsets for that fold (split into training and test set)
                # X_train_fold, y_train_fold = X2_train.iloc[train_idx], y2_train_mapped.iloc[train_idx]
                # X_test_fold, y_test_fold = X2_train.iloc[test_idx], y2_train_mapped.iloc[test_idx]
                
                cloned_pipeline.fit(X_train_fold, y_train_fold)
                
                # Check if both classes are in the test fold before scoring AUC
                if len(np.unique(y_test_fold)) > 1:
                    y_pred_proba = cloned_pipeline.predict_proba(X_test_fold)[:, 1]
                    # The parameter set with the best AUC score will be the chosen one for the second classifier
                    score = roc_auc_score(y_test_fold, y_pred_proba)
                    fold_scores.append(score)
            
            if fold_scores: # Only if we could calculate scores
                mean_score = np.mean(fold_scores) # append NaN when it fails
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
        
        print(f"Best AUC for Stage 2: {best_score:.4f}")
        print(f"Best Params for Stage 2: {best_params}")

        best_pipeline_c2 = clone(pipeline).set_params(**best_params)
        best_pipeline_c2.fit(X2_train, y2_train_mapped) # refit on all training data
        return best_pipeline_c2

        
    
    # def _perform_grid_search(self, pipeline, param_grid, X_train, y_train, groups):
    #     print("--- Starting Manual Grid Search Cross-Validation ---")
        
    #     cv = GroupKFold(n_splits=self.n_splits) # ensures that during CV the same group is not represented in both training and testing folds
    #     param_iterator = ParameterGrid(param_grid) # ParameterGrid(param_grid) ---> expands a dictionary of parameters into all possible combinations
        
    #     best_score = -1
    #     best_params = None

    #     # Loop that iterates through each parameter set
    #     for i, params in enumerate(param_iterator):
    #         #print(f"Testing params {i+1}/{len(param_iterator)}: {params}")
    #         fold_scores = []
            
    #         for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train, groups)): # cv.split() ---> splits the data into train/test indices according to groups
    #             # Create a fresh copy of the pipeline so that previous training doesn’t carry over
    #             cloned_pipeline = clone(pipeline)
    #             cloned_pipeline.set_params(**params) # inject the current parameter set into the pipeline
                
    #             # Use the fold indices to create training and test subsets for that fold (split into training and test set)
    #             X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    #             X_test_fold, y_test_fold = X_train.iloc[test_idx], y_train.iloc[test_idx]
                
    #             try:
    #                 cloned_pipeline.fit(X_train_fold, y_train_fold)
    #                 y_pred_proba = cloned_pipeline.predict_proba(X_test_fold)
    #                 score = roc_auc_score(y_test_fold, y_pred_proba, multi_class='ovr', average='weighted')
    #                 fold_scores.append(score)
    #             except Exception as e:
    #                 print(f"  [Fold {fold_idx+1}] Error during fit/score: {e}. Skipping fold.")
    #                 fold_scores.append(np.nan) # append NaN when it fails

    #         mean_score = np.nanmean(fold_scores) # np.nanmean ---> ignores failed foldsthat are NaN when computing the average score
    #         #print(f"  -> Mean CV Score (AUC): {mean_score:.4f}")

    #         if mean_score > best_score:
    #             best_score = mean_score
    #             best_params = params
        
    #     print("\n--- Manual Grid Search Results ---")
    #     print(f"Best Parameters Found: {best_params}")
    #     print(f"Best Mean CV Score (AUC): {best_score:.4f}")

    #     # Refit the best model on the entire training data
    #     print("Refitting the best model on the entire training set...")
    #     best_pipeline = clone(pipeline)
    #     best_pipeline.set_params(**best_params)
    #     best_pipeline.fit(X_train, y_train)
        
    #     return best_pipeline



    def evaluate(self, model_combo_name: str, feature_set_name: str, feature_list: list):
        print(f"\n--- Evaluating on Test Set ({model_combo_name}) ---")
        X_test = self.test_df[feature_list]
        y_true = self.test_df[self.classification_target]
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=[0, 1, 2], target_names=['No Hypo', 'Hypo', 'Deep Hypo'])
        auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')

        print("\n--- Multi-Class Classification Performance ---")
        print(f"Weighted OvR AUC-ROC: {auc_score:.4f}")
        print(classification_report(y_true, y_pred, target_names=['No Hypo', 'Hypo', 'Deep Hypo'], zero_division=0))
        
        self.results_log.append({
            'model_combination': model_combo_name,
            'feature_set': feature_set_name,
            'auc_roc_ovr_weighted': auc_score,
            'recall_hypo': report_dict['Hypo']['recall'],
            'recall_deep_hypo': report_dict['Deep Hypo']['recall'],
            'precision_hypo': report_dict['Hypo']['precision'],
            'precision_deep_hypo': report_dict['Deep Hypo']['precision'],
            'f1_weighted': report_dict['weighted avg']['f1-score']
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



    def run(self):
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

            X_train = self.train_df[feature_list]
            y_train = self.train_df[self.classification_target].astype(int)
            groups = self.train_df['subj_id']

            for model_combo_name, models in model_combinations.items():
                print(f"\n--- Testing Model Combination: {model_combo_name} ---")
                
                # Create separate pipelines for each stage
                pipeline_c1 = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', models['classifier1'])
                ])
               
                pipeline_c2 = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', models['classifier2'])
                ])

                # Create parameter grids with the 'model__' prefix for the pipeline
                params1_grid = {'model__' + k: v for k, v in models['params1'].items()} 
                params2_grid = {'model__' + k: v for k, v in models['params2'].items()}

                # STAGE 1 TUNING: Pass the pipeline and the corrected grid
                best_c1_pipeline = self._tune_classifier1(X_train, y_train, groups, pipeline_c1, params1_grid)
   
                # STAGE 2 TUNING: Pass the pipeline and the corrected grid
                best_c2_pipeline = self._tune_classifier2(X_train, y_train, groups, pipeline_c2, params2_grid)

                # Assemble the final model by extracting the best estimators from the tuned pipelines
                best_c1 = best_c1_pipeline.named_steps['model']
                best_c2 = best_c2_pipeline.named_steps['model']

                self.best_model = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', TwoStepClassifier(classifier1=best_c1, classifier2=best_c2))
                ])
               
                # Fit the final assembled pipeline on the full (unscaled) training data
                self.best_model.fit(X_train, y_train)
   
                # Evaluate the final assembled model
                self.evaluate(model_combo_name, feature_set_name, feature_list)

 
                
                
                
                
                
                