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

class ModelTrainer:
    """
    A class to handle the training of machine learning models and deep learning algorithms for 
    predicting postprandial glycemic events.
    """
    def __init__(self, data_file_path: str, analysis_type: str, 
                 classification_target: str, regression_target: str,
                 test_size: float = 0.2, random_state: int = 42):
        """
        Initializes the ModelTrainer.

        Args:
            data_file_path (str): Path to the CSV file with features.
            analysis_type (str): Mode of analysis, 'classification' or 'regression'.
            classification_target (str): Target column for classification.
            regression_target (str): Target column for regression.
            test_size (float): The proportion of the dataset to allocate to the test set.
            random_state (int): Seed for reproducibility of the random split.
        """
        self.data_file_path = data_file_path
        self.analysis_type = analysis_type
        self.test_size = test_size
        self.random_state = random_state

        if self.analysis_type == 'classification':
            self.target_column = classification_target
        elif self.analysis_type == 'regression':
            self.target_column = regression_target
        else:
            raise ValueError("analysis_type must be 'classification' or 'regression'")

        self.df = self._load_data()
        
        # These will be populated by the splitting method
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Placeholder for features to be used in the model
        self.features_for_prediction = []


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
        train_df = self.df[self.df['subj_id'].isin(train_subjects)].copy()
        test_df = self.df[self.df['subj_id'].isin(test_subjects)].copy()
        
        print("\n--- Data Split Summary (Greedy Method) ---")
        print(f"Total subjects: {len(subjects)}")
        print(f"Training subjects ({len(train_subjects)})")
        print(f"Test subjects ({len(test_subjects)})")
        print("-" * 25)
        print(f"Total postprandial periods: {len(self.df)}")
        print(f"Training periods: {len(train_df)} ({len(train_df) / len(self.df):.1%})")
        print(f"Test periods: {len(test_df)} ({len(test_df) / len(self.df):.1%})")
        print("-" * 25)

        # 6. Separate features (X) and target (y)
        # We will define the actual features to use later. For now, we prepare the split.
        # This step is currently a placeholder until we select the final features.
        # For now, let's assume we drop non-feature columns.
        non_feature_cols = [
            self.target_column, 'subj_id', 'meal_time_start', 'meal_time_end', 
            'postprandial_hypo_time', 'postprandial_deep_hypo_time', 'meal_category'
        ]
        all_numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.features_for_prediction = [
            col for col in all_numeric_cols if col not in non_feature_cols
        ]
        
        self.X_train = train_df[self.features_for_prediction]
        self.y_train = train_df[self.target_column]
        
        self.X_test = test_df[self.features_for_prediction]
        self.y_test = test_df[self.target_column]

        print(f"Features (X) shape for training: {self.X_train.shape}")
        print(f"Target (y) shape for training: {self.y_train.shape}")
        print(f"Features (X) shape for testing: {self.X_test.shape}")
        print(f"Target (y) shape for testing: {self.y_test.shape}")


    # def _split_data_by_subject(self):
    #     """
    #     Splits the data into training and test sets ensuring that subjects
    #     are unique to each set to prevent data leakage.
    #     """
    #     if self.df.empty or 'subj_id' not in self.df.columns:
    #         print("Cannot split data. DataFrame is empty or 'subj_id' column is missing.")
    #         return

    #     # 1. Get a unique list of all subject IDs
    #     all_subjects = self.df['subj_id'].unique()
        
    #     # 2. Split the list of subject IDs into training and testing lists
    #     train_subjects, test_subjects = train_test_split(
    #         all_subjects,
    #         test_size=self.test_size,
    #         random_state=self.random_state
    #     )

    #     # 3. Create the training and test DataFrames by filtering the original df
    #     train_df = self.df[self.df['subj_id'].isin(train_subjects)].copy()
    #     test_df = self.df[self.df['subj_id'].isin(test_subjects)].copy()

    #     print("\n--- Data Split Summary ---")
    #     print(f"Total subjects: {len(all_subjects)}")
    #     print(f"Training subjects ({len(train_subjects)}): {sorted(list(train_subjects))}")
    #     print(f"Test subjects ({len(test_subjects)}): {sorted(list(test_subjects))}")
    #     print("-" * 25)
    #     print(f"Total postprandial periods: {len(self.df)}")
    #     print(f"Training periods: {len(train_df)} ({len(train_df) / len(self.df):.1%})")
    #     print(f"Test periods: {len(test_df)} ({len(test_df) / len(self.df):.1%})")
    #     print("-" * 25)

    #     # 4. Separate features (X) from the target (y)
    #     # We will define the actual features to use later. For now, we prepare the split.
    #     # This step is currently a placeholder until we select the final features.
    #     # For now, let's assume we drop non-feature columns.
        
    #     # Define columns that are not features for the model
    #     non_feature_cols = [
    #         self.target_column, 'subj_id', 'meal_time_start', 'meal_time_end', 
    #         'postprandial_hypo_time', 'postprandial_deep_hypo_time', 'meal_category'
    #     ]
    #     # We will refine this list later
        
    #     # Get all numeric columns as a starting point, then drop the non-features
    #     all_numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
    #     self.features_for_prediction = [
    #         col for col in all_numeric_cols if col not in non_feature_cols
    #     ]
        
    #     # Create X and y sets
    #     self.X_train = train_df[self.features_for_prediction]
    #     self.y_train = train_df[self.target_column]
        
    #     self.X_test = test_df[self.features_for_prediction]
    #     self.y_test = test_df[self.target_column]

    #     print(f"Features (X) shape for training: {self.X_train.shape}")
    #     print(f"Target (y) shape for training: {self.y_train.shape}")
    #     print(f"Features (X) shape for testing: {self.X_test.shape}")
    #     print(f"Target (y) shape for testing: {self.y_test.shape}")


    def train(self):
        """
        Main orchestration method for the training pipeline.
        """
        if self.df.empty:
            return

        print("\n--- Starting Model Training Pipeline ---")
        
        # Step 1: Split the data
        self._split_data_by_subject()

        # Future steps will be added here:
        # Step 2: Preprocess the data (e.g., scaling, imputation, feature selection)
        # Step 3: Train and evaluate models (e.g., using cross-validation)
        # Step 4: Final evaluation on the held-out test set
        
        print("\nData splitting is complete. Ready for model implementation.")








# Example of how to run this class (for testing)
if __name__ == '__main__':
    trainer = ModelTrainer(
        data_file_path='postprandial_db/all_postprandial_data_with_features.csv',
        analysis_type='classification',
        classification_target='target_hypoglycemia',
        regression_target='postprandial_glucose_min'
    )
    trainer.train()