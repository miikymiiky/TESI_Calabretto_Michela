#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:27:42 2025

@author: michela
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def analyze_feature_outliers(
    features_db_path,
    data_dir,
    glucose_filename_pattern,
    top_n=10
):
    """
    Finds and plots the top N outliers for specified features to allow for
    visual inspection of the underlying raw data.
    """
    # --- 1. Load the feature database ---
    try:
        df = pd.read_csv(features_db_path)
        # Convert time columns to datetime objects
        for col in ['meal_time_start', 'meal_time_end']:
            df[col] = pd.to_datetime(df[col])
        print(f"Successfully loaded feature database from: {features_db_path}")
    except FileNotFoundError:
        print(f"ERROR: Feature database not found at {features_db_path}. Please run the feature extraction first.")
        return

    # --- 2. Define the features to check ---
    # We map the feature name to the type of raw data needed ('cgm' or 'steps')
    features_to_check = {
        'duration_hypo_minutes_24h_pre_meal': 'cgm',
        'duration_deep_hypo_minutes_24h_pre_meal': 'cgm',
        'duration_hyper_minutes_24h_pre_meal': 'cgm',
        'steps_during_rise': 'steps'
    }

    # --- 3. Loop through each feature, find outliers, and plot ---
    for feature, data_type in features_to_check.items():
        if feature not in df.columns:
            print(f"\nWarning: Feature '{feature}' not found in the database. Skipping.")
            continue

        print(f"\n{'='*60}\n--- Analyzing Top {top_n} Outliers for: {feature} ---\n{'='*60}")

        # Find the top N outliers for the current feature
        # We drop NA values to ensure we only get real calculated values
        top_outliers = df.dropna(subset=[feature]).nlargest(top_n, feature)

        if top_outliers.empty:
            print("No valid data found for this feature.")
            continue

        # Loop through each outlier and generate a plot
        for rank, (index, outlier_row) in enumerate(top_outliers.iterrows(), 1):
            
            subj_id = outlier_row['subj_id']
            meal_start = outlier_row['meal_time_start']
            meal_end = outlier_row['meal_time_end']
            feature_value = outlier_row[feature]

            # --- Load the appropriate raw data file ---
            raw_data_df = None
            plot_column = ''
            y_label = ''
            
            try:
                if data_type == 'cgm':
                    raw_file_path = os.path.join(data_dir, subj_id, glucose_filename_pattern.format(subj_id))
                    raw_data_df = pd.read_csv(raw_file_path, index_col='Time', parse_dates=True)
                    raw_data_df.sort_index(inplace=True)
                    plot_column = 'glucose_filtered'
                    y_label = 'Filtered Glucose (mg/dL)'
                
                elif data_type == 'steps':
                    raw_file_path = os.path.join(data_dir, subj_id, 'STEPS.csv')
                    raw_data_df = pd.read_csv(raw_file_path, header=None, names=['Time', 'STEPS'])
                    raw_data_df['STEPS'] = pd.to_numeric(raw_data_df['STEPS'], errors='coerce')
                    raw_data_df['Time'] = pd.to_datetime(raw_data_df['Time'], format='%d-%b-%Y %H:%M:%S', errors='coerce')
                    raw_data_df.dropna(subset=['Time'], inplace=True)
                    raw_data_df.set_index('Time', inplace=True)
                    raw_data_df.sort_index(inplace=True)
                    plot_column = 'STEPS'
                    y_label = 'Cumulative Steps'
            
            except FileNotFoundError:
                print(f"  - Could not plot for rank #{rank}: Raw data file not found at {raw_file_path}")
                continue
            
            if raw_data_df is None or plot_column not in raw_data_df.columns:
                 print(f"  - Could not plot for rank #{rank}: Column '{plot_column}' not in file.")
                 continue

            # --- Prepare for plotting ---
            fig, ax = plt.subplots(figsize=(15, 6))

            # Define the window for calculation to highlight on the plot
            calc_window_start, calc_window_end = None, None
            if 'pre_meal' in feature:
                calc_window_start = meal_start - pd.Timedelta(hours=24)
                calc_window_end = meal_start
                # Define a wider window for visualization context
                plot_start = meal_start - pd.Timedelta(hours=25)
                plot_end = meal_end
            elif 'steps_during_rise' in feature:
                time_to_peak = outlier_row.get('time_to_first_glucose_peak_after_meal', 0)
                if pd.isna(time_to_peak): time_to_peak = 0
                
                calc_window_start = meal_start
                calc_window_end = meal_start + pd.Timedelta(minutes=time_to_peak + 15)
                # Define a wider window for visualization context
                plot_start = meal_start - pd.Timedelta(hours=1)
                plot_end = meal_end + pd.Timedelta(hours=1)
            
            # Plot the raw data in the wider visualization window
            context_data = raw_data_df.loc[plot_start:plot_end]
            ax.plot(context_data.index, context_data[plot_column], marker='.', linestyle='-', label=f'Raw {data_type.upper()} Data')

            # Highlight the specific window used for the feature calculation
            if calc_window_start and calc_window_end:
                ax.axvspan(calc_window_start, calc_window_end, color='red', alpha=0.15, label=f'Calculation Window for Feature')
            
            # Add glycemic threshold lines
            if data_type == 'cgm':
                DEEP_HYPO_THRESHOLD = 54
                HYPO_THRESHOLD = 70
                HYPER_THRESHOLD = 130 # Based on the lambda function in your feature extractor

                if 'deep_hypo' in feature:
                    ax.axhline(y=DEEP_HYPO_THRESHOLD, color='darkred', linestyle='--', label=f'Deep Hypo Threshold ({DEEP_HYPO_THRESHOLD} mg/dL)')
                elif 'hypo' in feature:
                    ax.axhline(y=HYPO_THRESHOLD, color='orange', linestyle='--', label=f'Hypo Threshold ({HYPO_THRESHOLD} mg/dL)')
                    ax.axhline(y=DEEP_HYPO_THRESHOLD, color='darkred', linestyle=':', alpha=0.7, label=f'Deep Hypo Threshold ({DEEP_HYPO_THRESHOLD} mg/dL)')
                elif 'hyper' in feature:
                    ax.axhline(y=HYPER_THRESHOLD, color='purple', linestyle='--', label=f'Hyper Threshold ({HYPER_THRESHOLD} mg/dL)')

            # --- Formatting the plot ---
            ax.set_title(f"Outlier #{rank}: {feature}\nValue: {feature_value:.1f} | Subject: {subj_id} | Meal: {meal_start.strftime('%Y-%m-%d %H:%M')}", fontsize=14)
            ax.set_ylabel(y_label)
            ax.set_xlabel("Time")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            fig.autofmt_xdate() # Improve date formatting
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b %d'))
            
            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    DATABASE_DIR = "../postprandial_db"  # Go up one level from src
    DATA_DIR = "../data"                 # Go up one level from src
    FEATURES_FILENAME = "all_postprandial_data_with_features.csv"
    GLUCOSE_FILENAME_PATTERN = "{}tma_processed_new.csv"

    full_db_path = os.path.join(DATABASE_DIR, FEATURES_FILENAME)

    # Run the analysis
    analyze_feature_outliers(
        features_db_path=full_db_path,
        data_dir=DATA_DIR,
        glucose_filename_pattern=GLUCOSE_FILENAME_PATTERN,
        top_n=10
    )