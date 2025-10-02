#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 18:20:07 2025

@author: michela
"""

import pandas as pd
import numpy as np
import os
import json
import random
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, r2_score

# Model imports
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression



class HybridModelTrainer:
    """
    Trains and tunes a two-stage hybrid model using GridSearchCV.
    The model combinations are defined internally.
    """
    def __init__(self, data_file_path: str, features_file_path: str,
                 classification_target: str, regression_target: str,
                 test_size: float = 0.2, random_state: int = 42, n_splits: int = 5):
        
        self.data_file_path = data_file_path
        self.features_file_path = features_file_path
        self.classification_target = classification_target
        self.regression_target = regression_target
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = n_splits
        
        self.df = self._load_data()
        self.feature_sets = self._load_feature_sets()
        
        self.best_classifier = None
        self.best_regressor = None
        self.results_log = []



    def _get_model_combinations(self) -> dict:
        """
        Defines the dictionary of model combinations to be tested.
        This keeps the model configuration encapsulated within the class.
        """
        return {
            "XGB_Classifier_LGBM_Regressor": {
                "classifier": XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                "classifier_params": {
                    'model__n_estimators': [100, 200, 300], 
                    'model__max_depth': [3, 5, 7], 
                    'model__learning_rate': [0.01, 0.1],
                    'model__scale_pos_weight': [4, 4.5, 4.8, 5, 10, 15]
                },
                "regressor": LGBMRegressor(random_state=self.random_state, verbose=-1),
                "regressor_params": {
                    'model__n_estimators': [100, 200, 300], 
                    'model__num_leaves': [20, 31], 
                    'model__learning_rate': [0.01, 0.1]
                }
            },
            "XGB_Classifier_XGB_Regressor": {
                "classifier": XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                "classifier_params": {
                    'model__n_estimators': [100, 200, 300], 
                    'model__max_depth': [3, 5, 7], 
                    'model__learning_rate': [0.01, 0.1],
                    'model__scale_pos_weight': [4, 4.5, 4.8, 5, 10, 15]
                },
                "regressor": XGBRegressor(random_state=self.random_state),
                "regressor_params": {
                    'model__n_estimators': [100, 200, 300], 
                    'model__max_depth': [3, 5], 
                    'model__learning_rate': [0.01, 0.1]
                }
            },
            "LogReg_Classifier_LGBM_Regressor": {
                "classifier": LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
                "classifier_params": {
                    'model__penalty': ['l1', 'l2'], 
                    'model__C': [0.1, 1, 10], 
                    'model__solver': ['liblinear', 'saga']
                },
                "regressor": LGBMRegressor(random_state=self.random_state, verbose=-1),
                "regressor_params": {
                    'model__n_estimators': [100, 200, 300], 
                    'model__num_leaves': [20, 31], 
                    'model__learning_rate': [0.01, 0.1]
                }
            },
            "LogReg_Classifier_XGB_Regressor": {
                "classifier": LogisticRegression(random_state=self.random_state, max_iter=5000, class_weight='balanced'),
                "classifier_params": {
                    'model__penalty': ['l1', 'l2'], 
                    'model__C': [0.01, 0.05, 0.1, 0.5, 1.0, 10.0], 
                    'model__solver': ['liblinear', 'saga']
                },
                "regressor": XGBRegressor(random_state=self.random_state),
                "regressor_params": {
                    'model__n_estimators': [100, 200, 300], 
                    'model__max_depth': [1, 2, 3, 4, 5], 
                    'model__learning_rate': [0.01, 0.1, 0.3, 0.5]
                }
            }
        }



    def _load_data(self) -> pd.DataFrame:
        """Loads and prepares the feature DataFrame."""
        try:
            df = pd.read_csv(self.data_file_path)
            print(f"Successfully loaded data from {self.data_file_path}")
            # Drop rows where key targets are missing from the start
            return df.dropna(subset=[self.classification_target, self.regression_target])
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()



    def _load_feature_sets(self) -> dict:
        """Loads pre-selected feature sets from a JSON file."""
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



    def _perform_grid_search(self, pipeline, param_grid, X_train, y_train, groups, scoring, refit_metric):
        """Performs GridSearchCV and returns the best found estimator."""
        cv = GroupKFold(n_splits=self.n_splits)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, 
                                   scoring=scoring, refit=refit_metric, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train, groups=groups)
        
        print(f"Best Parameters (for {refit_metric}): {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_




    def train_and_tune(self, feature_list: list, classifier_model, classifier_params, regressor_model, regressor_params):
        """Tunes and trains both stages of the hybrid model."""
        print("\n--- Tuning and Training Hybrid Model ---")
        
        y_class_train = self.train_df[self.classification_target].apply(lambda x: 1 if x > 0 else 0) # Fusion of class 1 and 2 into the same class 1
        X_train = self.train_df[feature_list] # selection of the features to use for prediction
        groups = self.train_df['subj_id']

        classifier_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', classifier_model)
        ])
        
        print("\n>>> Tuning Classifier <<<")
        self.best_classifier = self._perform_grid_search(
            pipeline=classifier_pipeline, param_grid=classifier_params, X_train=X_train,
            y_train=y_class_train, groups=groups, scoring=['roc_auc', 'recall', 'average_precision'],
            refit_metric='recall'
            #refit_metric='roc_auc'
        )
        
        # Select only the Hypo events to train the regressor
        hypo_events_mask = self.train_df[self.classification_target] > 0
        train_df_hypo_only = self.train_df[hypo_events_mask]

        if train_df_hypo_only.empty:
            print("Warning: No hypoglycemic events in training set. Regressor cannot be trained.")
            self.best_regressor = None
            return

        y_reg_train = train_df_hypo_only[self.regression_target]
        X_reg_train = train_df_hypo_only[feature_list]
        groups_reg = train_df_hypo_only['subj_id']

        regressor_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', regressor_model)
        ])
        
        print("\n>>> Tuning Regressor <<<")
        self.best_regressor = self._perform_grid_search(
            pipeline=regressor_pipeline, param_grid=regressor_params, X_train=X_reg_train,
            y_train=y_reg_train, groups=groups_reg, scoring=['r2', 'neg_mean_absolute_error'],
            refit_metric='r2'
        )



    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions using the two-stage model."""
        class_preds = self.best_classifier.predict(X)
        final_preds = np.full(X.shape[0], 100.0) # generates an array filled with 100s
        hypo_indices = np.where(class_preds == 1)[0]
        
        if self.best_regressor and len(hypo_indices) > 0:
            X_to_predict_reg = X.iloc[hypo_indices] # select only the predicted 1s with the classifier
            reg_preds = self.best_regressor.predict(X_to_predict_reg)
            final_preds[hypo_indices] = reg_preds # replace the predicted 1s with the classifier, with the corresponding predicted glucose value with the regressor
            
        return final_preds



    def evaluate(self, feature_set_name: str, feature_list: list, model_combo_name: str):
        """Evaluates both parts of the model on the held-out test set."""
        print(f"\n--- Evaluating Hybrid Model on Test Set ({model_combo_name}) ---")
        X_test = self.test_df[feature_list]
        
        # 1. Component Evaluation (Separate) 
        # Classifier
        y_class_true = self.test_df[self.classification_target].apply(lambda x: 1 if x > 0 else 0)
        y_class_pred = self.best_classifier.predict(X_test)
        y_class_proba = self.best_classifier.predict_proba(X_test)[:, 1]
        class_auc = roc_auc_score(y_class_true, y_class_proba)
        report_dict = classification_report(y_class_true, y_class_pred, output_dict=True, zero_division=0, labels=[0, 1], target_names=['No Hypo', 'Hypo'])
        
        print("\n--- Classifier Performance ---")
        print(f"Classifier AUC-ROC: {class_auc:.4f}")
        print(classification_report(y_class_true, y_class_pred, target_names=['No Hypo', 'Hypo'], zero_division=0))
        
        # Regressor
        hypo_events_mask_test = self.test_df[self.classification_target] > 0
        test_df_hypo_only = self.test_df[hypo_events_mask_test]
        
        reg_mae, reg_r2 = np.nan, np.nan
        if self.best_regressor and not test_df_hypo_only.empty:
            y_reg_true = test_df_hypo_only[self.regression_target]
            X_reg_test = test_df_hypo_only[feature_list]
            y_reg_pred = self.best_regressor.predict(X_reg_test)
            reg_mae = mean_absolute_error(y_reg_true, y_reg_pred)
            reg_r2 = r2_score(y_reg_true, y_reg_pred)
            print("\n--- Regressor Performance (on true hypo events) ---")
            print(f"Mean Absolute Error (MAE): {reg_mae:.4f}")
            print(f"R-squared (R²):            {reg_r2:.4f}")
        else:
            print("\n--- Regressor Performance ---")
            print("No true hypoglycemic events in test set or regressor not trained. Skipping evaluation.")

        # 2. Overall System Evaluation (Combined) 
        # Get the final predictions from the entire hybrid system
        y_final_pred = self.predict(X_test)
        # Get the true minimum glucose values for ALL test samples
        y_reg_true_all = self.test_df[self.regression_target]
        
        # Create a mask to identify rows where the true value is not NaN
        valid_indices_mask = y_reg_true_all.notna()
        
        # Calculate the overall MAE
        # Apply the mask to both true and predicted values before calculating the error
        overall_system_mae = mean_absolute_error(y_reg_true_all[valid_indices_mask], y_final_pred[valid_indices_mask])
        print("\n--- Overall System Performance (Combined) ---")
        print(f"Overall System MAE: {overall_system_mae:.4f} mg/dL")

        # 3. Log the results 
        self.results_log.append({
            'model_combination': model_combo_name,
            'feature_set': feature_set_name,
            'overall_system_mae': overall_system_mae,
            'classifier_auc_roc': class_auc,
            'classifier_recall_hypo': report_dict['Hypo']['recall'],
            'classifier_precision_hypo': report_dict['Hypo']['precision'],
            'regressor_mae_on_hypo': reg_mae,
            'regressor_r2_on_hypo': reg_r2
        })



    def save_results_to_csv(self, filename="hybrid_performance_log.csv"):
        """Saves the collected experiment results to a CSV file."""
        if not self.results_log:
            print("No results to save.")
            return
        results_df = pd.DataFrame(self.results_log)
        db_directory = os.path.dirname(self.data_file_path)
        path = os.path.join(db_directory, filename)
        results_df.to_csv(path, index=False)
        print(f"\nSuccessfully saved hybrid model performance summary to: {path}")



    def run_pipeline(self):
        """Orchestrates the training and evaluation for all internally defined model combinations."""
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

            for model_combo_name, models in model_combinations.items():
                print(f"\n--- Testing Model Combination: {model_combo_name} ---")
                
                self.train_and_tune(
                    feature_list=feature_list, classifier_model=models['classifier'],
                    classifier_params=models['classifier_params'], regressor_model=models['regressor'],
                    regressor_params=models['regressor_params']
                )
                self.evaluate(feature_set_name, feature_list, model_combo_name)









if __name__ == '__main__':
    print("--- Running Hybrid Model Trainer in standalone test mode ---")
    trainer = HybridModelTrainer(
        data_file_path='postprandial_db/all_postprandial_data_with_features.csv',
        features_file_path='postprandial_db/classification_selected_feature_sets.json',
        classification_target='target_hypoglycemia',
        regression_target='postprandial_glucose_min',
        random_state=42
    )
    trainer.run_pipeline()
    trainer.save_results_to_csv("tuned_hybrid_performance_log.csv")