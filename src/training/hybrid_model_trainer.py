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
#from imblearn.over_sampling import SMOTE
#from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, r2_score

# Model imports
#from xgboost import XGBClassifier, XGBRegressor
#from lightgbm import LGBMRegressor
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
        
        initial_rows = len(self.df)
        # Keep only rows where the regression target is greater than 0
        self.df = self.df[self.df[self.regression_target] > 0].copy()
        rows_dropped = initial_rows - len(self.df)
        if rows_dropped > 0:
            print(f"Removed {rows_dropped} row(s) with negative target values.")
                
        self.df = self._apply_pre_split_filters(self.df) # Apply filters before splitting
        self.feature_sets = self._load_feature_sets()
        self._display_loaded_feature_sets()
        
        self.best_classifier = None
        self.best_regressor = None
        self.results_log = []
    
    
    
    
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
        Defines the dictionary of model combinations to be tested.
        This keeps the model configuration encapsulated within the class.
        """
        return {
            "XGB_Classifier_LGBM_Regressor": {
                "classifier": XGBClassifier(random_state=self.random_state, eval_metric='logloss', scale_pos_weight=5.0),
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
                "classifier": XGBClassifier(random_state=self.random_state, eval_metric='logloss', scale_pos_weight=5.0),
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




    def train_and_tune(self, X_train, y_class_train, y_reg_train, groups, feature_list, classifier_model, classifier_params, regressor_model, regressor_params, use_smote=False):
        """Tunes and trains both stages of the hybrid model on pre-cleaned data."""
        print("\n--- Tuning and Training Hybrid Model ---")

        classifier_pipeline_steps = [
            # ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]

        if use_smote:
            print(">>> SMOTE enabled for the classifier stage.")
            classifier_pipeline_steps.insert(1, ('sampler', SMOTE(random_state=self.random_state)))
        
        classifier_pipeline_steps.append(('model', classifier_model))
        
        classifier_pipeline = ImbPipeline(classifier_pipeline_steps)
        
        
        # --- CHANGES START ---
        
        print("\n>>> Fitting Classifier on full training data <<<")
        self.best_classifier = classifier_pipeline.fit(X_train, y_class_train)
        
        
        
        # TUNING & CV TO COMMENT OUT FOR NOW
        # print("\n>>> Tuning Classifier <<<")
        # self.best_classifier = self._perform_grid_search(
        #     pipeline=classifier_pipeline, param_grid=classifier_params, X_train=X_train,
        #     y_train=y_class_train, groups=groups, scoring=['roc_auc', 'recall', 'average_precision'],
        #     refit_metric='recall'
        #     #refit_metric='roc_auc'
        # )
        
        # --- CHANGES FINISH ---
        
        
        
        # Select only the Hypo events from the already cleaned data to train the regressor
        hypo_events_mask = y_class_train > 0
        if not hypo_events_mask.any():
            print("Warning: No hypoglycemic events in the cleaned training set. Regressor cannot be trained.")
            self.best_regressor = None
            return

        y_reg_train_hypo = y_reg_train[hypo_events_mask]
        X_reg_train_hypo = X_train[hypo_events_mask]
        groups_reg = groups[hypo_events_mask]

        regressor_pipeline = Pipeline([
            # ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', regressor_model)
        ])
        
        
        
        # --- CHANGES START ---
        
        print("\n>>> Fitting Regressor on hypo-only training data <<<")
        self.best_regressor = regressor_pipeline.fit(X_reg_train_hypo, y_reg_train_hypo)



        # TUNING & CV TO COMMENT OUT FOR NOW
        # print("\n>>> Tuning Regressor <<<")
        # self.best_regressor = self._perform_grid_search(
        #     pipeline=regressor_pipeline, param_grid=regressor_params, X_train=X_reg_train,
        #     y_train=y_reg_train, groups=groups_reg, scoring=['r2', 'neg_mean_absolute_error'],
        #     refit_metric='r2'
        # )
        
        # --- CHANGES FINISH



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



    def evaluate(self, X_test, y_class_true, y_reg_true, test_df_full, feature_set_name: str, model_combo_name: str):
        """Evaluates both parts of the model on the pre-cleaned test set."""
        print(f"\n--- Evaluating Hybrid Model on Test Set ({model_combo_name}) ---")
        
        # 1. Component Evaluation (Separate) 
        # Classifier
        y_class_pred = self.best_classifier.predict(X_test)
        y_class_proba = self.best_classifier.predict_proba(X_test)[:, 1]
        class_auc = roc_auc_score(y_class_true, y_class_proba)
        report_dict = classification_report(y_class_true, y_class_pred, output_dict=True, zero_division=0, labels=[0, 1], target_names=['No Hypo', 'Hypo'])
        
        print("\n--- Classifier Performance ---")
        print(f"Classifier AUC-ROC: {class_auc:.4f}")
        print(classification_report(y_class_true, y_class_pred, target_names=['No Hypo', 'Hypo'], zero_division=0))
        
        # Time Gain Calculation
        true_positives_mask = (y_class_true.values > 0) & (y_class_pred > 0)
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
            
        # Regressor
        hypo_events_mask_test = y_class_true > 0
        
        reg_mae, reg_r2 = np.nan, np.nan
        mean_prediction_horizon, median_prediction_horizon = np.nan, np.nan
        
        if self.best_regressor and hypo_events_mask_test.any():
            y_reg_true_hypo = y_reg_true[hypo_events_mask_test]
            X_reg_test_hypo = X_test[hypo_events_mask_test]
            
            y_reg_pred = self.best_regressor.predict(X_reg_test_hypo)
            reg_mae = mean_absolute_error(y_reg_true_hypo, y_reg_pred)
            reg_r2 = r2_score(y_reg_true_hypo, y_reg_pred)
            print("\n--- Regressor Performance (on true hypo events) ---")
            print(f"Mean Absolute Error (MAE): {reg_mae:.4f}")
            print(f"R-squared (R²):            {reg_r2:.4f}")
            
            # --- Prediction Horizon Analysis ---
            print("\n--- Regressor Prediction Horizon Analysis ---")
            analysis_df = test_df_full.loc[hypo_events_mask_test].copy()
            
            hypo_time_series = self.df['postprandial_hypo_time']
            analysis_df = analysis_df.merge(hypo_time_series, left_index=True, right_index=True, how='left')
            
            analysis_df['meal_time_start'] = pd.to_datetime(analysis_df['meal_time_start'])
            analysis_df['postprandial_hypo_time'] = pd.to_datetime(analysis_df['postprandial_hypo_time'])

            prediction_time = analysis_df['meal_time_start'] + pd.to_timedelta(analysis_df['time_to_first_glucose_peak_after_meal'] + 15, unit='m')
            prediction_horizon = (analysis_df['postprandial_hypo_time'] - prediction_time).dt.total_seconds() / 60
            prediction_horizon = prediction_horizon[prediction_horizon > 0].dropna()
            
            if not prediction_horizon.empty:
                mean_prediction_horizon = prediction_horizon.mean()
                median_prediction_horizon = prediction_horizon.median()
                print(f"Number of hypo events with valid horizon: {len(prediction_horizon)}")
                print(f"Average Prediction Horizon: {mean_prediction_horizon:.1f} minutes")
                print(f"Median Prediction Horizon:  {median_prediction_horizon:.1f} minutes")
            else:
                print("No valid prediction horizons could be calculated.")
                
        else:
            print("\n--- Regressor Performance ---")
            print("No true hypoglycemic events in test set or regressor not trained. Skipping evaluation.")

        # 2. Overall System Evaluation (Combined) 
        y_final_pred = self.predict(X_test)
        
        # Ensure y_reg_true is aligned with X_test for the final MAE calculation
        valid_indices_mask = y_reg_true.notna()
        overall_system_mae = mean_absolute_error(y_reg_true[valid_indices_mask], y_final_pred[valid_indices_mask])
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
            'mean_time_gain_min': mean_time_gain,
            'median_time_gain_min': median_time_gain,
            'regressor_mae_on_hypo': reg_mae,
            'regressor_r2_on_hypo': reg_r2,
            'reg_mean_prediction_horizon': mean_prediction_horizon,
            'reg_median_prediction_horizon': median_prediction_horizon,
            'best_params': f"CLF: {self.best_classifier.get_params()['model']} | REG: {self.best_regressor.get_params()['model'] if self.best_regressor else 'N/A'}"
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



    def run_pipeline(self, use_smote=False):
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
            
            
            # --- Handle NaNs for both train and test sets ---
            # Training set
            X_train_initial = self.train_df[feature_list]
            print("Final class distribution in training set (for classifier):") # Print the original distribution first
            print(self.train_df[self.classification_target].value_counts())
            y_class_train_initial = self.train_df[self.classification_target].apply(lambda x: 1 if x > 0 else 0)
            # Add the required time columns for the Time Gain calculation
            train_full = pd.concat([X_train_initial, y_class_train_initial, self.train_df[self.regression_target], self.train_df['subj_id'], self.train_df['meal_time_start'], self.train_df['time_to_first_glucose_peak_after_meal']], axis=1)            
            initial_train_rows = len(train_full)
            train_full.dropna(inplace=True)

            print(f"Dropped {initial_train_rows - len(train_full)} rows with NaN values from the training set.")
            print(f"Final training set dimensions: {train_full.shape[0]} postprandial periods.")
            print("Final class distribution in training set (for classifier):")
            print(train_full[self.classification_target].value_counts())

            # Test set
            X_test_initial = self.test_df[feature_list]
            print("Final class distribution in test set (for classifier):") # Print the original distribution first
            print(self.test_df[self.classification_target].value_counts())
            y_class_test_initial = self.test_df[self.classification_target].apply(lambda x: 1 if x > 0 else 0)
            # Add the required time columns for the Time Gain calculation
            test_full = pd.concat([X_test_initial, y_class_test_initial, self.test_df[self.regression_target], self.test_df['subj_id'], self.test_df['meal_time_start'], self.test_df['time_to_first_glucose_peak_after_meal']], axis=1)
            initial_test_rows = len(test_full)
            test_full.dropna(inplace=True)

            print(f"\nDropped {initial_test_rows - len(test_full)} rows with NaN values from the test set.")
            print(f"Final test set dimensions: {test_full.shape[0]} postprandial periods.")
            print("Final class distribution in test set (for classifier):")
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
            y_class_train = train_full[self.classification_target]
            y_reg_train = train_full[self.regression_target]
            groups = train_full['subj_id']

            X_test = test_full[feature_list]
            y_class_test = test_full[self.classification_target]
            y_reg_test = test_full[self.regression_target]
            

            for model_combo_name, models in model_combinations.items():
                print(f"\n--- Testing Model Combination: {model_combo_name} ---")
                
                self.train_and_tune(
                    X_train=X_train, y_class_train=y_class_train, y_reg_train=y_reg_train, groups=groups,
                    feature_list=feature_list, classifier_model=models['classifier'],
                    classifier_params=models['classifier_params'], regressor_model=models['regressor'],
                    regressor_params=models['regressor_params'],
                    use_smote=use_smote 
                )
                self.evaluate(X_test, y_class_test, y_reg_test, test_full, feature_set_name, model_combo_name)
        
        # Conditionally name the log file and save the results
        log_filename = "tuned_hybrid_performance_log.csv"
        if use_smote:
            log_filename = "tuned_hybrid_performance_log_with_smote.csv"
        
        self.save_results_to_csv(log_filename)









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