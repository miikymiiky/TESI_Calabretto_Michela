#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:18:54 2025

@author: michela
"""

import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalyzer:
    """
    Analyzes and visualizes performance logs from various model training runs.
    """
    def __init__(self, logs_directory: str):
        self.logs_directory = logs_directory
        self.master_df = self._load_and_merge_logs()
        self.analysis_type = "Unknown" # Default value
        
        if not self.master_df.empty:
            print(f"Successfully loaded and merged {len(self.master_df)} results from {logs_directory}")
            # Detect analysis type
            if 'roc_auc_ovr_weighted' in self.master_df.columns or 'auc_roc' in self.master_df.columns:
                self.analysis_type = "Classification"
            elif 'overall_system_mae' in self.master_df.columns:
                self.analysis_type = "Hybrid"
            elif 'r2' in self.master_df.columns:
                self.analysis_type = "Regression"



    def _get_log_files(self):
        """Helper to find all relevant performance log files."""
        # This pattern finds both standard and smote logs
        return glob.glob(os.path.join(self.logs_directory, "*_performance_log*.csv")) # glob.glob() ---> finds all file paths matching a specific pattern



    def _load_and_merge_logs(self) -> pd.DataFrame:
        log_files = self._get_log_files()
        if not log_files:
            print("Warning: No performance logs found in the specified directory.")
            return pd.DataFrame()
        
        df_list = []
        for f in log_files:
            try:
                df = pd.read_csv(f)
                filename = os.path.basename(f)
                
                # Add a column to track if SMOTE was used
                df['smote_used'] = 'with_smote' in filename
                
                # Add a column for the analysis mode by parsing the filename
                if 'binary_classification' in filename:
                    df['analysis_mode'] = 'Binary Classification'
                elif 'multi_class_classification' in filename:
                    df['analysis_mode'] = 'Multi-Class Classification'
                elif 'hybrid' in filename:
                    df['analysis_mode'] = 'Hybrid'
                elif 'two_step' in filename:
                    df['analysis_mode'] = 'Two-Step'
                elif 'regression' in filename:
                    df['analysis_mode'] = 'Regression'
                else:
                    df['analysis_mode'] = 'Unknown'
                    
                df_list.append(df)
            except Exception as e:
                print(f"Warning: Could not read or process log file {f}. Error: {e}")
        
        if not df_list:
            return pd.DataFrame()
            
        return pd.concat(df_list, ignore_index=True)
    
    
    
    def display_log_summary(self):
        """
        Groups results by analysis mode and prints a detailed, formatted summary 
        for every model/feature set combination found in the logs.
        """
        if self.master_df.empty:
            print("No performance data loaded. Cannot generate summary.")
            return

        grouped = self.master_df.groupby(['analysis_mode', 'smote_used'])

        for (mode, smote_status), group_df in grouped:
            
            metric, ascending = None, False
            if mode == 'Hybrid':
                metric, ascending = 'overall_system_mae', True
            elif mode in ['Multi-Class Classification', 'Two-Step']:
                metric = 'roc_auc_ovr_weighted'
            elif mode == 'Binary Classification':
                metric = 'auc_roc'
            elif mode == 'Regression':
                metric = 'r2'

            print("\n" + "="*80)
            print(f"### FULL RESULTS FOR: {mode} (SMOTE: {smote_status}) ###")
            print("="*80)

            if group_df.empty:
                print("No data found for this group.")
                continue
            
            sorted_df = group_df.copy()
            if metric and metric in sorted_df.columns and sorted_df[metric].notna().any():
                sorted_df = sorted_df.sort_values(by=['feature_set', metric], ascending=[True, ascending])
            
            for _, row in sorted_df.iterrows():
                model_name = row.get('model_combination') or row.get('model', 'N/A')
                feature_set = row.get('feature_set', 'N/A')

                print(f"\n--- Model: {model_name} | Feature Set: {feature_set} ---")

                if mode == 'Binary Classification':
                    print(f"  - Accuracy     : {row.get('accuracy', 'N/A'):.4f}")
                    print(f"  - Precision    : {row.get('precision', 'N/A'):.4f}")
                    print(f"  - Recall       : {row.get('recall', 'N/A'):.4f}")
                    print(f"  - Specificity  : {row.get('specificity', 'N/A'):.4f}")
                    print(f"  - F1-Score     : {row.get('f1_score', 'N/A'):.4f}")
                    print(f"  - MCC          : {row.get('mcc', 'N/A'):.4f}")
                    print(f"  - AUC-ROC      : {row.get('auc_roc', 'N/A'):.4f}")
                    print(f"  - AUC-PR       : {row.get('auc_pr', 'N/A'):.4f}")

                elif mode == 'Multi-Class Classification':
                    print(f"  - Accuracy     : {row.get('accuracy', 'N/A'):.4f}")
                    print(f"  - Precision (W): {row.get('precision_weighted', 'N/A'):.4f}")
                    print(f"  - Recall (W)   : {row.get('recall_weighted', 'N/A'):.4f}")
                    print(f"  - F1-Score (W) : {row.get('f1_weighted', 'N/A'):.4f}")
                    print(f"  - MCC          : {row.get('mcc', 'N/A'):.4f}")
                    print(f"  - AUC-ROC (OvR): {row.get('roc_auc_ovr_weighted', 'N/A'):.4f}")
                    print(f"  - Specificity (No Hypo) : {row.get('specificity_no_hypo', 'N/A'):.4f}")
                    print(f"  - Specificity (Hypo)    : {row.get('specificity_hypo', 'N/A'):.4f}")
                    print(f"  - Specificity (Deep Hypo): {row.get('specificity_deep_hypo', 'N/A'):.4f}")

                elif mode == 'Two-Step':
                    print(f"  - AUC-ROC (OvR): {row.get('auc_roc_ovr_weighted', 'N/A'):.4f}")
                    print(f"  - F1-Score (W) : {row.get('f1_weighted', 'N/A'):.4f}")
                    print(f"  - Recall (Hypo)    : {row.get('recall_hypo', 'N/A'):.4f}")
                    print(f"  - Recall (Deep Hypo): {row.get('recall_deep_hypo', 'N/A'):.4f}")
                    print(f"  - Precision (Hypo)    : {row.get('precision_hypo', 'N/A'):.4f}")
                    print(f"  - Precision (Deep Hypo): {row.get('precision_deep_hypo', 'N/A'):.4f}")

                elif mode == 'Hybrid':
                    print(f"  - Overall System MAE   : {row.get('overall_system_mae', 'N/A'):.4f}")
                    print(f"  - Classifier AUC       : {row.get('classifier_auc_roc', 'N/A'):.4f}")
                    print(f"  - Classifier Recall    : {row.get('classifier_recall_hypo', 'N/A'):.4f}")
                    print(f"  - Classifier Precision : {row.get('classifier_precision_hypo', 'N/A'):.4f}")
                    print(f"  - Regressor R² (on Hypo): {row.get('regressor_r2_on_hypo', 'N/A'):.4f}")
                    print(f"  - Regressor MAE (on Hypo): {row.get('regressor_mae_on_hypo', 'N/A'):.4f}")

                elif mode == 'Regression':
                    print(f"  - R-squared (R²): {row.get('r2', 'N/A'):.4f}")
                    print(f"  - MAE           : {row.get('mae', 'N/A'):.4f}")
                    print(f"  - RMSE          : {row.get('rmse', 'N/A'):.4f}")
                
                print(f"  - Best Parameters : {row.get('best_params', 'N/A')}")
        print("\n" + "="*80)


        
    def display_best_models(self, sort_by='roc_auc_ovr_weighted', top_n=1):
        if self.master_df.empty or sort_by not in self.master_df.columns:
            print(f"Cannot display summary. Metric '{sort_by}' not found in logs.")
            return
        # Ascending order if sorting by MAE, descending order otherwise
        ascending = True if 'mae' in sort_by.lower() else False
        best_idx = self.master_df.groupby('feature_set')[sort_by].idxmax() # .groupby('feature_set') ---> groups rows by the feature set column
                                                                           # [sort_by] ---> selects the metric column to evaluate
                                                                           # .idxmax() ---> for each group, returns the row index of the maximum value
        
        # Select the rows corresponding to the best model for each feature set
        summary_df = self.master_df.loc[best_idx].sort_values(by=sort_by, ascending=ascending)
        
        display_cols = ['feature_set', 'model_combination', 'model', sort_by, 'recall_weighted', 'precision_weighted', 'overall_system_mae']
        existing_cols = [col for col in display_cols if col in summary_df.columns]
        
        print(summary_df[existing_cols].to_string(index=False))



    def plot_metric_comparison(self, metric='roc_auc_ovr_weighted', title=None):
        if self.master_df.empty or metric not in self.master_df.columns:
            print(f"Cannot plot metric '{metric}'. Column not found.")
            return
            
        plt.figure(figsize=(14, 8))
        
        df = self.master_df.dropna(subset=[metric]).copy()
        # Use a simpler model label for the legend
        df['model_label'] = df['model_combination'].fillna(df['model'])

        sns.barplot(x='feature_set', y=metric, hue='model_label', data=df, palette='viridis')
        
        if title is None:
            title = f"{self.analysis_type} Model Comparison: {metric.replace('_', ' ').title()}"
        
        plt.title(title, fontsize=16)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xlabel('Feature Set')
        plt.xticks(rotation=0) # No rotation needed with few x-axis items
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left') # legend at the top left corner
        plt.tight_layout()
        plt.show()



    def plot_precision_recall_tradeoff(self):
        if 'recall_weighted' in self.master_df.columns and 'precision_weighted' in self.master_df.columns:
            recall_col, precision_col = 'recall_weighted', 'precision_weighted'
        elif 'classifier_recall_hypo' in self.master_df.columns and 'classifier_precision_hypo' in self.master_df.columns:
            recall_col, precision_col = 'classifier_recall_hypo', 'classifier_precision_hypo'
        else:
            print("Could not find suitable precision/recall columns for trade-off plot.")
            return
            
        plt.figure(figsize=(12, 8))
        
        plot_df = self.master_df.dropna(subset=[recall_col, precision_col]).copy()
        plot_df['model_label'] = plot_df['model_combination'].fillna(plot_df['model'])
        if plot_df.empty:
            print("No data available to plot precision-recall tradeoff.")
            return

        sns.scatterplot(
            data=plot_df, x=precision_col, y=recall_col, hue='feature_set',
            style='model_label', s=150, alpha=0.8
        )
        
        # Dynamic title for this plot as well
        plt.title(f'{self.analysis_type} Precision vs. Recall Trade-off', fontsize=16)
        plt.xlabel('Precision')
        plt.ylabel('Recall (Sensitivity)')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Model & Feature Set', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
  
        
  
    def analyze_feature_sets(self, metric='roc_auc_ovr_weighted'):
        if self.master_df.empty or metric not in self.master_df.columns:
            print(f"Cannot analyze feature sets by metric '{metric}'. Column not found.")
            return
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='feature_set', y=metric, data=self.master_df, palette='pastel')
        sns.stripplot(x='feature_set', y=metric, data=self.master_df, color='black', alpha=0.3)
        
        # Dynamic title here too
        plt.title(f'{self.analysis_type} Performance Distribution by Feature Set', fontsize=16)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xlabel('Feature Set')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()