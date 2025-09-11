#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 18:41:01 2025

@author: michela
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def visualize_interpolated_events(
    features_db_path: str,
    data_dir: str,
    glucose_filename_pattern: str,
    max_plots: int = 50
):
    """
    Finds and plots all meal events where saturation interpolation was applied.
    """
    try:
        df = pd.read_csv(features_db_path)
        for col in ['meal_time_start', 'meal_time_end']:
            df[col] = pd.to_datetime(df[col])
        print(f"Successfully loaded feature database from: {features_db_path}")
    except FileNotFoundError:
        print(f"ERROR: Feature database not found at '{features_db_path}'.")
        return

    interpolated_df = df[df['was_interpolated'] == 1].copy()
    total_found = len(interpolated_df)

    if total_found == 0:
        print("No interpolated events found in the database. Nothing to visualize.")
        return

    print(f"Found {total_found} interpolated meal events to visualize.")

    plots_to_show = total_found
    if total_found > max_plots:
        user_input = input(f"Display all {total_found} plots? (This may take a while). "
                           f"Enter 'yes' to proceed or anything else to show only the first {max_plots}: ").lower()
        if user_input != 'yes':
            plots_to_show = max_plots
            print(f"Showing the first {max_plots} plots...")
    
    for index, row in interpolated_df.head(plots_to_show).iterrows():
        subj_id = row['subj_id']
        meal_start = row['meal_time_start']
        meal_end = row['meal_time_end']

        glucose_file_path = os.path.join(data_dir, subj_id, glucose_filename_pattern.format(subj_id))
        try:
            glucose_df = pd.read_csv(glucose_file_path, index_col='Time', parse_dates=True)
        except FileNotFoundError:
            print(f"⚠️ Warning: Could not find glucose file for subject {subj_id}. Skipping.")
            continue


        # Load and plot the specific columns

        # Check if the final column exists. If not, the feature extraction hasn't been run yet.
        if 'glucose_final_for_features' not in glucose_df.columns:
            print(f"ERROR: Column 'glucose_final_for_features' not found in {glucose_file_path}.")
            print("Please run the feature extraction step in main.py first.")
            continue

        # Get the series directly from the file
        glucose_series_raw = glucose_df['glucose']
        glucose_series_filtered_original = glucose_df['glucose_filtered']
        glucose_series_final = glucose_df['glucose_final_for_features']

        plot_start = meal_start - pd.Timedelta(hours=1)
        plot_end = meal_end + pd.Timedelta(hours=1)

        # Slice the series for plotting
        glucose_raw_window = glucose_series_raw.loc[plot_start:plot_end]
        glucose_filtered_original_window = glucose_series_filtered_original.loc[plot_start:plot_end]
        glucose_final_window = glucose_series_final.loc[plot_start:plot_end]
    
        changed_points_mask = (glucose_filtered_original_window != glucose_final_window)
        interpolated_segment = glucose_final_window[changed_points_mask]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(glucose_raw_window.index, glucose_raw_window, 
                color='lightgray', marker='x', markersize=4, linestyle=':', 
                label='Raw Glucose')

        ax.plot(glucose_filtered_original_window.index, glucose_filtered_original_window, 
                color='dodgerblue', marker='o', markersize=4, linestyle='-', 
                label='Original Filtered Glucose')

        ax.plot(glucose_final_window.index, glucose_final_window, 
                color='red', linestyle='--', linewidth=2,
                label='Interpolated Glucose (Final)')
        
        if not interpolated_segment.empty:
            ax.scatter(interpolated_segment.index, interpolated_segment.values,
                       color='red', s=80, zorder=5, label='Interpolated Points')

        ax.axvline(meal_start, color='green', linestyle='--', label=f'Meal Start ({meal_start.strftime("%H:%M")})')
        ax.axvline(meal_end, color='purple', linestyle=':', label=f'Meal End ({meal_end.strftime("%H:%M")})')
        
        # Add a horizontal line for the saturation threshold 
        ax.axhline(y=39, color='black', linestyle='-.', linewidth=1, alpha=0.7, label='Saturation Threshold (39 mg/dL)')

        ax.set_title(f"Interpolation Inspection\nSubject: {subj_id} | Meal Start: {meal_start.strftime('%Y-%m-%d %H:%M')}", fontsize=16)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Glucose (mg/dL)", fontsize=12)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    project_root = os.path.join(script_dir, '..')
    
    DATABASE_DIR = os.path.join(project_root, "postprandial_db")
    DATA_DIR = os.path.join(project_root, "data")
    FEATURES_FILENAME = "all_postprandial_data_with_features.csv"
    GLUCOSE_FILENAME_PATTERN = "{}tma_processed_new.csv"

    full_db_path = os.path.join(DATABASE_DIR, FEATURES_FILENAME)

    # Run the Visualization
    visualize_interpolated_events(
        features_db_path=full_db_path,
        data_dir=DATA_DIR,
        glucose_filename_pattern=GLUCOSE_FILENAME_PATTERN,
        max_plots=50  # this default limit can be changed
    )