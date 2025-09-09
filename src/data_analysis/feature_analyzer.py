# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:15:58 2025

@author: prend
"""

# src/data_analysis/feature_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Union, List, Dict, Any
from scipy.stats import pearsonr # for calculating Pearson correlation
from scipy.stats import kruskal # libraries for statistical tests
import pingouin as pg
import scikit_posthocs as sp # alternative library to pingouin

pd.set_option('display.max_columns', None) # to display all columns of the results tables, no matter how wide the table is
pd.set_option('display.width', 1000) # Set a wide display width to prevent line wrapping


class FeatureAnalyzer:
    """
    Classe per l'analisi delle feature estratte e analisi rispetto al target.
    """
    def __init__(self, data_file_path, analysis_type='classification', 
                 classification_target='target_hypoglycemia', 
                 regression_target='postprandial_glucose_min',
                 feature_set='all'):
        """
        Inizializza l'analizzatore di feature.

        Args:
            data_file_path (str): Path to the CSV file with features.
            analysis_type (str): Mode of analysis, either 'classification' or 'regression'.
            classification_target (str): Name of the target column for classification.
            regression_target (str): Name of the target column for regression.
            feature_set (str): Which feature set to use: 'all', 'activity', or 'no_activity'.
        """
        self.data_file_path = data_file_path
        self.analysis_type = analysis_type
        self.classification_target = classification_target
        self.regression_target = regression_target
        self.feature_set = feature_set
        
        
        self.df = self._load_data()
        self.numeric_cols = []
        
        # Define target label mapping
        self.target_label_map = { # dictionary with the hypo classes
            0: 'No Hypo',
            1: 'Hypo',
            2: 'Deep Hypo'
        }
        
        
        if self.df is not None:
            # Set the target column based on the analysis type
            if self.analysis_type == 'classification':
                self.target_column = self.classification_target
                if self.target_column not in self.df.columns:
                    print(f"Warning: Classification target '{self.target_column}' not found.")
                else:
                    # Create labeled column for plotting
                    self.df[f'{self.target_column}_labeled'] = self.df[self.target_column].map(self.target_label_map).fillna('Unknown')
                    numeric_target_values = self.df[self.target_column].dropna().unique()
                    ordered_labels = [self.target_label_map[int(i)] for i in sorted(numeric_target_values) if int(i) in self.target_label_map]
                    self.df[f'{self.target_column}_labeled'] = pd.Categorical(self.df[f'{self.target_column}_labeled'], categories=ordered_labels, ordered=True)
                    self.df[self.target_column] = self.df[self.target_column].astype('category')
            
            elif self.analysis_type == 'regression':
                self.target_column = self.regression_target
                if self.target_column not in self.df.columns:
                    print(f"Warning: Regression target '{self.target_column}' not found.")
            else:
                raise ValueError("analysis_type must be 'classification' or 'regression'")

            # Select numeric features for analysis
            self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            if 'subj_id' in self.numeric_cols: self.numeric_cols.remove('subj_id')
            if self.target_column in self.numeric_cols: self.numeric_cols.remove(self.target_column)


        self.features_to_exclude = [
            "postprandial_duration_minutes",
            "has_battery_status_data",
            "has_heart_rate_data",
            "has_heart_rate_variability_data",
            "has_pulse_ox_data",
            "has_respiration_data",
            "has_steps_data",
            "has_zero_crossing_data",
            "has_zero_crossing_energytotal_data",
            "has_zero_crossing_timeelapsed_data",
            "postprandial_hypo",
            "postprandial_hypo_duration_minutes",
            "postprandial_deep_hypo",
            "postprandial_deep_hypo_duration_minutes",
            "glucose_at_first_peak",
            "hypo_to_check_flag",
            "is_saturated_at_39"
        ]
        
        
        

    def _load_data(self):
        """Carica il DataFrame delle feature."""
        if not os.path.exists(self.data_file_path):
            print(f"Errore: File '{self.data_file_path}' non trovato.")
            return None
        try:
            df = pd.read_csv(self.data_file_path)
            # Converte le colonne di tempo se necessario (se caricate come stringhe)
            for col in ['meal_time_start', 'meal_time_end', 'postprandial_hypo_time', 'postprandial_deep_hypo_time']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Caricato il DataFrame da {self.data_file_path} con {len(df)} righe e {len(df.columns)} colonne.")
            return df
        except Exception as e:
            print(f"Errore durante il caricamento del file {self.data_file_path}: {e}")
            return None
        
    
    def _get_features_for_analysis(self):
        """
        Restituisce la lista di feature numeriche da analizzare,
        escludendo quelle specificate e filtrando per il set di feature selezionato.
        """
        # Define keywords that identify physical activity features
        activity_keywords = ['steps_', 'activity_', 'cadence_', 'light_', 'moderate_', 'vigorous_']

        # Get all base analyzable features
        all_features = [f for f in self.numeric_cols if f not in self.features_to_exclude]

        if self.feature_set == 'activity':
            print("--- Analyzing ONLY Physical Activity Features ---")
            return [f for f in all_features if any(keyword in f for keyword in activity_keywords)]
        
        elif self.feature_set == 'no_activity':
            print("--- Analyzing all features EXCEPT Physical Activity ---")
            return [f for f in all_features if not any(keyword in f for keyword in activity_keywords)]
        
        else: # Default to 'all'
            print("--- Analyzing ALL Features ---")
            return all_features



    def display_basic_info(self):
        """Mostra informazioni di base sul DataFrame."""
        if self.df is None: return
        print("\n--- Informazioni di Base sul DataFrame ---")
        print(self.df.info(verbose=False))
        # print("\n--- Statistiche Descrittive delle Feature Numeriche ---")
        # print(self.df[self.numeric_cols].describe())
        
        print(f"\n--- Distribuzione del Target ('{self.target_column}') ---")
        if self.analysis_type == 'classification':
            print(self.df[f'{self.target_column}_labeled'].value_counts(dropna=False))
        else: # For regression, show descriptive stats
            print(self.df[self.target_column].describe())
        
        # # Optional: Print distribution of labeled target for verification
        # if f'{self.target_column}_labeled' in self.df.columns:
        #     print(f"\n--- Distribuzione del Target Etichettato ('{self.target_column}_labeled') ---")
        #     print(self.df[f'{self.target_column}_labeled'].value_counts(dropna=False))

        print("\n--- Percentuale di Valori Mancanti per Colonna ---")
        missing_percentage = self.df.isnull().sum() * 100 / len(self.df)
        print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))
        
        
    def filter_data(self, conditions: Union[Dict[str, Any], List[Dict[str, Any]]]): # type hint that specifies that the conditions argument can be either:
                                                                                    # - a single dictionary (Dict[str, Any]), where keys are strings and values can be of any type
                                                                                    # - a list of such dictionaries (List[Dict[str, Any]])
        """
        Filtra il DataFrame delle feature in base a una o più condizioni.
        Il DataFrame interno alla classe (self.df) viene aggiornato con le righe filtrate.

        Args:
            conditions (Union[Dict[str, Any], List[Dict[str, Any]]]): 
                Una singola condizione o una lista di condizioni.
                Ogni condizione deve essere un dizionario con le chiavi:
                'column' (str): Nome della colonna.
                'operator' (str): Operatore di confronto (es. '>', '<', '==', '>=', '<=', '!=').
                'value' (Any): Valore da confrontare.
                
                Se viene fornita una lista, le condizioni sono combinate con un'operazione AND.
        """
        if self.df is None:
            print("DataFrame non caricato. Impossibile applicare il filtro.")
            return

        # Normalizza l'input a una lista di condizioni anche se è una singola condizione
        if not isinstance(conditions, list):
            conditions = [conditions]

        original_rows = len(self.df)
        
        # Inizializza una maschera booleana con tutti True, che verrà aggiornata con AND
        combined_mask = pd.Series(True, index=self.df.index)

        for condition in conditions:
            column = condition.get('column')
            operator = condition.get('operator')
            value = condition.get('value')

            # Controlla che tutti i campi necessari siano presenti
            if not all([column, operator, value is not None]):
                print(f"Errore: Una condizione non è formattata correttamente. Richiede 'column', 'operator' e 'value'. Condizione ignorata: {condition}")
                continue # Salta questa condizione e prosegui con le altre

            if column not in self.df.columns:
                print(f"Avviso: Colonna '{column}' non trovata nel DataFrame. Questa condizione verrà ignorata.")
                continue # Salta questa condizione

            current_mask = None
            try:
                if operator == '>':
                    current_mask = self.df[column] > value
                elif operator == '<':
                    current_mask = self.df[column] < value
                elif operator == '==':
                    current_mask = self.df[column] == value
                elif operator == '>=':
                    current_mask = self.df[column] >= value
                elif operator == '<=':
                    current_mask = self.df[column] <= value
                elif operator == '!=':
                    current_mask = self.df[column] != value
                else:
                    print(f"Avviso: Operatore '{operator}' non supportato per la colonna '{column}'. Operatori validi: '>', '<', '==', '>=', '<=', '!='. Questa condizione verrà ignorata.")
                    continue
            except TypeError as e:
                print(f"Errore di tipo durante il filtraggio della colonna '{column}' con operatore '{operator}' e valore '{value}': {e}. Assicurati che i tipi di dato siano compatibili.")
                continue # Salta questa condizione in caso di errore di tipo
            except Exception as e:
                print(f"Si è verificato un errore inatteso durante il filtraggio: {e}. Condizione ignorata.")
                continue
            
            # Combina le maschere con un'operazione AND
            combined_mask = combined_mask & current_mask
        
        self.df = self.df[combined_mask]
        print(f"Filtro applicato con {len(conditions)} condizioni. "
              f"Righe filtrate da {original_rows} a {len(self.df)}.")
        
        # Dopo il filtraggio, potremmo voler ri-aggiornare numeric_cols se il filtraggio
        # ha rimosso tutte le righe per determinate colonne, ma per questo caso non è strettamente necessario.
        # Rimangono comunque le stesse colonne, solo meno righe.

        
    def plot_correlations(self, figsize=(16, 12), output_path=None):
        """
        Genera una heatmap della matrice di correlazione tra le feature numeriche.
        """
        if self.df is None or not self.numeric_cols:
            print("Nessun dato o colonne numeriche disponibili per la correlazione.")
            return

        print("\n--- Calcolo Matrice di Correlazione ---")
        features_for_analysis = self._get_features_for_analysis()
        correlation_matrix = self.df[features_for_analysis].corr()

        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.05) # sns = seaborn
                                                                                                 # annot = False ---> doesn't annotate/display the correlation values
                                                                                                 # fmt=".2f" ---> specifies the format string for the annotations (if annot = True)
        plt.title('Matrice di Correlazione', fontsize=16)
       
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Heatmap di correlazione salvata a: {output_path}")
        plt.show()
    
    
    # Correlation plot with the new features
    def plot_focused_correlation(self, feature_list, title='Focused Correlation Heatmap', figsize=(12, 10), output_path=None, annotate_cells=True):
        """
        Generates a correlation heatmap for a specific subset of features.
        """
        valid_features = [f for f in feature_list if f in self.df.columns]
        if len(valid_features) < 2:
            print(f"Warning: Not enough valid features found in the DataFrame to plot '{title}'.")
            return

        correlation_matrix = self.df[valid_features].corr()

        plt.figure(figsize=figsize)
        # Set annot=True to see the correlation values
        sns.heatmap(correlation_matrix, annot=annotate_cells, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Focused heatmap saved to: {output_path}")
        plt.show()
        

    def plot_feature_distributions(self, num_cols_per_plot=5, figsize=(15, 4)):
        """
        Genera istogrammi per mostrare la distribuzione di ogni feature numerica.
        """
        
        features_for_analysis = self._get_features_for_analysis()
            
        print("\n--- Plotting Distribuzioni delle Feature ---")
        for i in range(0, len(features_for_analysis), num_cols_per_plot):
            cols_to_plot = features_for_analysis[i:i + num_cols_per_plot]
            fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(figsize[0], figsize[1])) # figsize = (figsize[0], figsize[1]) ---> sets the size of the overall figure
            if len(cols_to_plot) == 1: # Handle single subplot case
                axes = [axes]
            
            for j, col in enumerate(cols_to_plot):
                sns.histplot(self.df[col].dropna(), kde=True, ax=axes[j]) # kde = True ---> tells histplot to overlay a Kernel Density Estimate (KDE) curve on top of the histogram
                                                                          # the KDE provides a smoothed representation of the data's distribution
                axes[j].set_title(f'Distribuzione di {col}')
                axes[j].set_xlabel(col)
                axes[j].set_ylabel('Frequenza')
            plt.tight_layout()
            plt.show()
            
    
    def display_regression_correlations(self, top_n=25): # top_n=25 ---> optional parameter that determines how many top correlated features to show (default 25)
        """
        Calculates and displays Pearson correlations of features with the regression target.
        """
        if self.analysis_type != 'regression':
            print("This function is only available for regression analysis.")
            return
        
        if self.df is None or self.target_column not in self.df.columns:
            print("DataFrame or regression target column not available.")
            return
        
        
        print(f"\n--- Statistical Significance of Features for Regression Target ('{self.target_column}') ---")
        
        # Use the same exclusion logic as the classification tests to avoid data leakage
        features_to_exclude_for_prediction = [
            'target_hypoglycemia',
            'time_peak_hypo_event',
            'postprandial_glucose_min', # This is the target itself
            'second_glucose_peak_value_after_meal',
            'time_to_second_glucose_peak_after_meal',
        ]
        for col in self.df.columns:
            if any(keyword in col for keyword in ['postprandial_', '_until_nadir', '_later_post_peak']):
                if col not in features_to_exclude_for_prediction:
                    features_to_exclude_for_prediction.append(col)

        all_exclusions = self.features_to_exclude + features_to_exclude_for_prediction
        features_to_test = [f for f in self._get_features_for_analysis() if f not in all_exclusions]

        results = []
        target = self.target_column

        for feature in features_to_test:
            # Drop NaNs pairwise for correlation
            temp_df = self.df[[feature, target]].dropna()
            if len(temp_df) < 3: # Need at least 3 data points
                continue
            
            try:
                corr, pval = pearsonr(temp_df[feature], temp_df[target])
                results.append({'Feature': feature, 'Pearson_r': corr, 'P-Value': pval})
            except Exception as e:
                print(f"Could not calculate correlation for '{feature}': {e}")
        
        if not results:
            print("No correlation results were generated.")
            return
            
        # Create a DataFrame and sort by p-value
        results_df = pd.DataFrame(results).sort_values(by='P-Value').reset_index(drop=True)

        # Print the top N features
        print(f"\n--- Top {top_n} Most Significant Features for Regression ---")
        print(results_df.head(top_n))

        # Print all features
        print("\n--- All Significant Features for Regression (Sorted by P-Value) ---")
        with pd.option_context('display.max_rows', None): # Temporarily display all rows
            print(results_df)
        
        # Save results to CSV 
        output_path = 'regression_feature_significance.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\nFull regression results saved to: {output_path}")

        return results_df, features_to_test

    
    def plot_distributions_by_target(self, num_cols_per_plot=3, figsize=(18, 6)):
        """
        Generates plots to show feature distributions in relation to the target.
        - Classification: Box plots.
        - Regression: Scatter plots with a regression line.
        """
        if self.df is None or self.target_column not in self.df.columns or not self.numeric_cols:
            print("Nessun dato, colonna target o colonne numeriche disponibili per l'analisi.")
            return

        features_for_analysis = self._get_features_for_analysis()

        if self.analysis_type == 'classification':
            # CLASSIFICATION PLOTS (BOXPLOTS)
            print(f"\n--- Plotting Distributions per Categoria di Target ({self.target_column}) ---")
            target_col_for_plot = f'{self.target_column}_labeled'
            custom_palette_for_strings = {'No Hypo': '#C8E6C9', 'Hypo': '#FFECB3', 'Deep Hypo': '#FFCDD2'}
            
            for i in range(0, len(features_for_analysis), num_cols_per_plot):
                cols_to_plot = features_for_analysis[i:i + num_cols_per_plot]
                fig, axes = plt.subplots(1, len(cols_to_plot), figsize=figsize)
                if len(cols_to_plot) == 1: axes = [axes]

                for j, col in enumerate(cols_to_plot):
                    ax = axes[j]
                    if self.df[col].notna().any():
                        sns.boxplot(x=target_col_for_plot, y=col, data=self.df, ax=ax, palette=custom_palette_for_strings, 
                                    order=self.df[target_col_for_plot].cat.categories, hue=target_col_for_plot, legend=False)
                        sns.stripplot(x=target_col_for_plot, y=col, data=self.df, ax=ax, color='black', size=3, alpha=0.1, 
                                      jitter=0.2, order=self.df[target_col_for_plot].cat.categories)
                        ax.set_title(f'{col}')
                        ax.set_xlabel('')
                    else:
                        ax.set_title(f'{col}')
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                plt.tight_layout()
                plt.show()

        elif self.analysis_type == 'regression':
            # REGRESSION PLOTS (SCATTER/REGPLOTS) 
            print(f"\n--- Plotting Feature vs. Target ({self.target_column}) ---")
            for i in range(0, len(features_for_analysis), num_cols_per_plot):
                cols_to_plot = features_for_analysis[i:i + num_cols_per_plot]
                fig, axes = plt.subplots(1, len(cols_to_plot), figsize=figsize)
                if len(cols_to_plot) == 1: axes = [axes]

                for j, col in enumerate(cols_to_plot):
                    ax = axes[j]
                    plot_data = self.df[[col, self.target_column]].dropna()
                    
                    if not plot_data.empty and len(plot_data) > 1:
                        sns.regplot(x=col, y=self.target_column, data=plot_data, ax=ax,
                                    scatter_kws={'alpha': 0.3, 's': 20, 'color': 'navy'}, 
                                    line_kws={'color': 'red', 'linewidth': 2})
                        
                        r, p = pearsonr(plot_data[col], plot_data[self.target_column])
                        title = f'{col}\nR={r:.3f}, p={p:.3f}'
                        if p < 0.001: title = f'{col}\nR={r:.3f}, p<0.001'
                        ax.set_title(title)
                    else:
                        ax.set_title(col)
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                plt.tight_layout()
                plt.show()
                
    
    
    def plot_postmeal_steps_by_meal_category(self):
       if self.df is None or 'meal_category' not in self.df.columns:
           return
       
       feature_to_plot = 'steps_during_rise'
       if feature_to_plot not in self.df.columns:
           return

       print("\n--- Plotting Post-Meal Steps (During Rise) by Meal Category ---")
       meal_categories = sorted(self.df['meal_category'].dropna().unique())
       
       # This plot is more complex for regression, so we'll adapt its behavior
       if self.analysis_type == 'classification':
           target_col_for_plot = f'{self.target_column}_labeled'
           custom_palette = {'No Hypo': '#C8E6C9', 'Hypo': '#FFECB3', 'Deep Hypo': '#FFCDD2'}
           
           fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
           fig.suptitle('Post-Meal Steps (During Rise) by Meal Category and Hypo Event', fontsize=18)

           for ax, category in zip(axes.flatten(), meal_categories):
               meal_df = self.df[self.df['meal_category'] == category]
               if meal_df.empty or meal_df[feature_to_plot].isnull().all():
                   ax.set_title(f'{category} (No Data)')
                   continue
               
               sns.boxplot(x=target_col_for_plot, y=feature_to_plot, data=meal_df, ax=ax, palette=custom_palette,
                           order=self.df[target_col_for_plot].cat.categories, hue=target_col_for_plot, legend=False)
               sns.stripplot(x=target_col_for_plot, y=feature_to_plot, data=meal_df, ax=ax, color='black', 
                             size=3, alpha=0.15, jitter=0.2, order=self.df[target_col_for_plot].cat.categories)
               ax.set_title(f'{category} Meals')
               ax.set_xlabel('')
               ax.set_ylabel('Steps Taken During Rise')
           plt.tight_layout(rect=[0, 0, 1, 0.96])
           plt.show()

       elif self.analysis_type == 'regression':
           # For regression, we can't group by hypo class. A simpler violin plot is more appropriate.
           plt.figure(figsize=(10, 6))
           sns.violinplot(x='meal_category', y=feature_to_plot, data=self.df, order=meal_categories, palette='pastel')
           plt.title('Post-Meal Steps (During Rise) Distribution by Meal Category')
           plt.xlabel('Meal Category')
           plt.ylabel('Steps Taken During Rise')
           plt.grid(axis='y', linestyle='--', alpha=0.7)
           plt.show()
           
    
    
    def plot_steps_by_meal_category(self):
        if self.df is None or 'meal_category' not in self.df.columns:
            return

        print("\n--- Plotting Pre-Meal Steps by Meal Category ---")
        step_features = [f'steps_{h}h_pre_meal' for h in [1, 2, 3, 4]]
        meal_categories = sorted(self.df['meal_category'].dropna().unique())

        if self.analysis_type == 'classification':
            target_col_for_plot = f'{self.target_column}_labeled'
            custom_palette = {'No Hypo': '#C8E6C9', 'Hypo': '#FFECB3', 'Deep Hypo': '#FFCDD2'}
            
            for category in meal_categories:
                meal_df = self.df[self.df['meal_category'] == category]
                if meal_df.empty: continue
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle(f'Pre-Meal Steps for {category} Meals by Hypo Event', fontsize=18)
                
                for ax, feature in zip(axes.flatten(), step_features):
                    if meal_df[feature].notna().any():
                        sns.boxplot(x=target_col_for_plot, y=feature, data=meal_df, ax=ax, palette=custom_palette,
                                    order=self.df[target_col_for_plot].cat.categories, hue=target_col_for_plot, legend=False)
                        sns.stripplot(x=target_col_for_plot, y=feature, data=meal_df, ax=ax, color='black', size=3, alpha=0.15, jitter=0.2,
                                      order=self.df[target_col_for_plot].cat.categories)
                        ax.set_title(f'Distribution of {feature}')
                        ax.set_xlabel('')
                        ax.set_ylabel('Steps Taken')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()

        elif self.analysis_type == 'regression':
            # For regression, we'll plot the distributions of pre-meal steps for each meal category.
            df_melted = self.df.melt(id_vars=['meal_category'], value_vars=step_features, 
                                     var_name='Time Window', value_name='Steps')
            
            plt.figure(figsize=(12, 7))
            sns.boxplot(x='meal_category', y='Steps', hue='Time Window', data=df_melted, 
                        order=meal_categories, palette='viridis')
            plt.title('Pre-Meal Steps Distribution by Meal Category and Time Window')
            plt.xlabel('Meal Category')
            plt.ylabel('Steps Taken')
            plt.legend(title='Time Window Pre-Meal')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    
    
    def run_classification_tests(self):
        if self.analysis_type != 'classification':
            print("This function is designed for the classification task.")
            return

        print("\n--- Running Statistical Tests for Classification Feature Selection ---")
        
        non_normal_features = [
            'time_since_previous_meal_minutes','time_to_first_glucose_peak_after_meal',
            'meal_hour','meal_day_of_week',
        ]
        non_normal_keywords = [
            'num_', 'number_', 'prop_', 'proportion', 'duration_', 
            'lbgi', 'hbgi', 'time_from_', 'time_since_', 'time_to_', '_x_',
            'MAGE', 'J_index', 'bgri', 'ADRR', 'ROC'
        ]
        for col in self.df.columns:
            if any(keyword in col for keyword in non_normal_keywords):
                if col not in non_normal_features:
                    non_normal_features.append(col)

        features_to_exclude_for_prediction = [
            'target_hypoglycemia','time_peak_hypo_event','postprandial_glucose_min',
            'second_glucose_peak_value_after_meal','time_to_second_glucose_peak_after_meal',
        ]
        for col in self.df.columns:
            if any(keyword in col for keyword in ['postprandial_', '_until_nadir', '_later_post_peak']):
                if col not in features_to_exclude_for_prediction:
                    features_to_exclude_for_prediction.append(col)
        
        all_exclusions = self.features_to_exclude + features_to_exclude_for_prediction
        features_to_test = [f for f in self._get_features_for_analysis() if f not in all_exclusions]
        
        results = []
        target = self.target_column
        
        for feature in features_to_test:
            temp_df = self.df[[feature, target]].dropna()
            if temp_df.empty or temp_df[target].nunique() < 2:
                continue

            test_used, stat, pval, effect_size = "", np.nan, np.nan, np.nan

            try:
                if feature in non_normal_features:
                    test_used = "Kruskal-Wallis"
                    # Use pingouin's kruskal to get effect size (epsilon-squared)
                    kruskal_res = pg.kruskal(data=temp_df, dv=feature, between=target)
                    
                    # Check if the result is a valid DataFrame before accessing
                    if isinstance(kruskal_res, pd.DataFrame) and not kruskal_res.empty:
                        stat = kruskal_res['H'].iloc[0]
                        pval = kruskal_res['p-unc'].iloc[0]
                        # Safely get effect size, as 'eps' may not always be present
                        effect_size = kruskal_res['eps'].iloc[0] if 'eps' in kruskal_res.columns else np.nan
                
                else:
                    test_used = "Welch's ANOVA"
                    # Use Welch's ANOVA for the robust F-stat and p-value
                    aov = pg.welch_anova(dv=feature, between=target, data=temp_df)
                    
                    # Also check here for Welch's ANOVA
                    if isinstance(aov, pd.DataFrame) and not aov.empty:
                        stat = aov['F'].iloc[0]
                        pval = aov['p-unc'].iloc[0]
                
                        # Effect size calculation also needs protection
                        aov_effect = pg.anova(dv=feature, between=target, data=temp_df)
                        if isinstance(aov_effect, pd.DataFrame) and 'np2' in aov_effect.columns:
                            effect_size = aov_effect['np2'].iloc[0]
                        else:
                            effect_size = np.nan
                
                # Only append results if the test was successfully performed
                if pd.notna(pval):
                    results.append({
                        'Feature': feature, 'Test': test_used, 'Statistic': stat, 
                        'P-Value': pval, 'Effect_Size': effect_size
                })

            except Exception as e:
                print(f"Could not perform test for feature '{feature}': {e}")

        if not results:
            print("No statistical test results were generated.")
            return
        
        # Add the new Effect_Size column to the DataFrame
        results_df = pd.DataFrame(results).sort_values(by='P-Value').reset_index(drop=True)
        
        print("\n--- Top 25 Most Significant Features for Classification ---")
        print(results_df.head(25))
        
        print("\n--- All Significant Features for Classification (Sorted by P-Value) ---")
        with pd.option_context('display.max_rows', None):
            print(results_df)
        
        output_path = 'classification_omnibus_tests.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\nFull omnibus test results saved to: {output_path}")
        
        return results_df, features_to_test
    
    
    
    def run_posthoc_tests(self, significant_features_df):
        """
        Performs pairwise post-hoc tests using a hybrid approach for stability.
        - Pingouin for Games-Howell (Welch's ANOVA followup).
        - Scikit-posthocs for Dunn's test (Kruskal-Wallis followup).
        """
        if self.analysis_type != 'classification':
            print("This function is designed for the classification task.")
            return

        significant_features_df = significant_features_df[significant_features_df['P-Value'] < 0.05].copy()

        if significant_features_df.empty:
            print("No features with p < 0.05 were found to run post-hoc tests.")
            return

        print("\n--- Running Pairwise Post-Hoc Tests for Significant Features (p < 0.05) ---")
        
        posthoc_results = []
        target = self.target_column
        
        for _, row in significant_features_df.iterrows():
            feature = row['Feature']
            omnibus_test = row['Test']

            if self.df[feature].isnull().all():
                continue
            
            try:
                if omnibus_test == "Welch's ANOVA":
                    # Use Pingouin for Games-Howell
                    pairwise_df = pg.pairwise_gameshowell(data=self.df, dv=feature, between=target)
                    test_name = "Games-Howell"
                    
                    # Parse Pingouin's DataFrame format
                    for _, posthoc_row in pairwise_df.iterrows():
                        group1 = self.target_label_map.get(posthoc_row['A'], str(posthoc_row['A']))
                        group2 = self.target_label_map.get(posthoc_row['B'], str(posthoc_row['B']))
                        stat_val = posthoc_row.get('T', np.nan)
                        pval = posthoc_row.get('pval', np.nan)

                        posthoc_results.append({
                            'Feature': feature, 'Comparison': f"{group1} vs. {group2}",
                            'Test': test_name, 'Statistic': stat_val, 'P-Value': pval
                        })

                elif omnibus_test == "Kruskal-Wallis":
                    # Use scikit-posthocs for Dunn's test
                    # The observed=False argument silences the FutureWarning
                    grouped_data = [group[feature].dropna() for _, group in self.df.groupby(target, observed=False)]
                    p_values = sp.posthoc_dunn(grouped_data, p_adjust='holm')
                    test_name = "Dunn"
                    
                    # Parse scikit-posthocs's matrix format
                    group_indices = sorted(self.df[target].dropna().unique())
                    p_values.columns = group_indices
                    p_values.index = group_indices

                    for i in range(len(group_indices)):
                        for j in range(i + 1, len(group_indices)):
                            group1_idx = group_indices[i]
                            group2_idx = group_indices[j]
                            group1_label = self.target_label_map.get(group1_idx)
                            group2_label = self.target_label_map.get(group2_idx)
                            
                            posthoc_results.append({
                                'Feature': feature, 'Comparison': f"{group1_label} vs. {group2_label}",
                                'Test': test_name, 'Statistic': np.nan, 'P-Value': p_values.loc[group1_idx, group2_idx]
                            })
            except Exception as e:
                print(f"Could not perform post-hoc test for feature '{feature}': {e}")
                
        if not posthoc_results:
            print("No post-hoc test results were generated.")
            return
        
        # Create the final, separate DataFrame from the collected results
        results_df = pd.DataFrame(posthoc_results)

        
        print("\n--- Pairwise Comparisons for Top 25 Most Significant Features ---")
        # Filter the detailed results to show only the top 25 features
        top_25_features = significant_features_df['Feature'].head(25).tolist()
        top_25_posthoc_df = results_df[results_df['Feature'].isin(top_25_features)].copy()
        # Ensure the feature order is maintained for the printout
        top_25_posthoc_df['Feature'] = pd.Categorical(top_25_posthoc_df['Feature'], categories=top_25_features, ordered=True)
        print(top_25_posthoc_df.sort_values('Feature'))

        print("\n--- All Significant Pairwise Comparisons (Ordered by Omnibus Test Significance) ---")
        with pd.option_context('display.max_rows', None):
            print(results_df)
        
        # Save results to CSV
        output_path = 'classification_posthoc_tests.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\nFull post-hoc test results saved to: {output_path}")
            
        return results_df
    
    
    def get_pairwise_comparison_results(self, significant_features_df):
        """
        Performs post-hoc tests and returns a 'wide' DataFrame with pairwise p-values in columns.
        This is a helper function for advanced filtering and does not print to the console.
        """
        significant_df = significant_features_df[significant_features_df['P-Value'] < 0.05].copy()
        if significant_df.empty:
            return pd.DataFrame()

        # Define column names for the wide DataFrame
        group_labels = {
            0: self.target_label_map.get(0, "Group 0"),
            1: self.target_label_map.get(1, "Group 1"),
            2: self.target_label_map.get(2, "Group 2")
        }
        comparisons = {
            (0, 1): f"P-Value ({group_labels[0]} vs {group_labels[1]})",
            (0, 2): f"P-Value ({group_labels[0]} vs {group_labels[2]})",
            (1, 2): f"P-Value ({group_labels[1]} vs {group_labels[2]})"
        }
        for col_name in comparisons.values():
            significant_df[col_name] = np.nan

        target = self.target_column
        
        # Iterate through features and fill the p-value columns
        for index, row in significant_df.iterrows():
            feature = row['Feature']
            omnibus_test = row['Test']

            if self.df[feature].isnull().all():
                continue
            
            try:
                if omnibus_test == "Welch's ANOVA":
                    pairwise_df = pg.pairwise_gameshowell(data=self.df, dv=feature, between=target)
                    for _, posthoc_row in pairwise_df.iterrows():
                        pair = tuple(sorted((int(posthoc_row['A']), int(posthoc_row['B']))))
                        if pair in comparisons:
                            col_name = comparisons[pair]
                            significant_df.at[index, col_name] = posthoc_row.get('pval', np.nan)
                
                elif omnibus_test == "Kruskal-Wallis":
                    grouped_data = [group[feature].dropna() for _, group in self.df.groupby(target, observed=False)]
                    p_values = sp.posthoc_dunn(grouped_data, p_adjust='holm')
                    group_indices = sorted(self.df[target].dropna().unique())
                    p_values.columns = group_indices
                    p_values.index = group_indices
                    for group_pair, col_name in comparisons.items():
                        g1, g2 = group_pair
                        if g1 in p_values.index and g2 in p_values.columns:
                            significant_df.at[index, col_name] = p_values.loc[g1, g2]
            except Exception:
                # Silently fail for this helper, the main function will report errors
                continue
                
        # Return only the feature name and the new p-value columns
        return significant_df[['Feature'] + list(comparisons.values())]
    
    
    def select_uncorrelated_features(self, significant_features_df, top_n=35, threshold=0.8):
        """
        Selects a set of top features with low correlation among them.

        Args:
            significant_features_df (pd.DataFrame): DataFrame of features sorted by significance.
            top_n (int): The desired number of less-correlated features to return.
            threshold (float): The correlation threshold to use for exclusion.

        Returns:
            list: A list of feature names.
        """
        if self.df is None:
            print("DataFrame not loaded.")
            return []

        # Get the complete list of features, ordered by their original significance
        feature_list = significant_features_df['Feature'].tolist()
        
        # Calculate the correlation matrix for these features
        correlation_matrix = self.df[feature_list].corr().abs()
        
        uncorrelated_features = []
        
        for feature in feature_list:
            # Always add the very first feature
            if not uncorrelated_features:
                uncorrelated_features.append(feature)
                continue
            
            # Check correlation against features already selected
            is_correlated = False
            for selected_feature in uncorrelated_features:
                if correlation_matrix.loc[feature, selected_feature] > threshold:
                    is_correlated = True
                    break # Stop checking if it's correlated with any one of them
            
            # If it's not correlated with any of the selected features, add it
            if not is_correlated:
                uncorrelated_features.append(feature)
            
            # Stop when we have enough features
            if len(uncorrelated_features) >= top_n:
                break
                
        print(f"\nSelected {len(uncorrelated_features)} features with correlation less than {threshold}.")
        return uncorrelated_features
            










    def analyze_clusters(self, features_for_clustering=None, num_clusters=3, plot_type='scatter'):
        """
        Tenta di visualizzare i dati per suggerire la presenza di cluster.
        Questo metodo non esegue un clustering formale, ma visualizza in 2D/3D.
        Per un clustering formale, servirebbero algoritmi come K-Means/DBSCAN.
        """
        if self.df is None or not self.numeric_cols:
            print("Nessun dato o colonne numeriche disponibili per l'analisi dei cluster.")
            return

        print("\n--- Analisi Preliminare per Cluster ---")

        if features_for_clustering is None:
            # Prendi alcune delle feature più informative, o le prime due/tre se non specificate
            # Potresti voler selezionare feature con alta varianza o poca correlazione
            selected_features = self.numeric_cols[:min(3, len(self.numeric_cols))] # min(3, len(self.numeric_cols))---> takes the min between 3 and len(self.numeric_cols)
            if not selected_features:
                print("Nessuna feature numerica disponibile per la visualizzazione dei cluster.")
                return
            print(f"Visualizzazione delle feature: {', '.join(selected_features)}")
        else:
            selected_features = [f for f in features_for_clustering if f in self.numeric_cols]
            if len(selected_features) < 2:
                print("Selezionare almeno due feature numeriche per la visualizzazione dei cluster.")
                return

        # Use the labeled target column for plotting
        target_col_for_plot = f'{self.target_column}_labeled'
        if target_col_for_plot not in self.df.columns:
            print(f"Errore: Colonna etichettata '{target_col_for_plot}' non trovata per il plotting.")
            return

        df_plot = self.df.dropna(subset=selected_features + [target_col_for_plot])
        
        if df_plot.empty:
            print("Nessun dato valido per le feature selezionate dopo aver rimosso i NaN.")
            return

        # Define custom palette for target labels using string labels as keys
        custom_palette_for_strings = {
            'No Hypo': 'skyblue',
            'Hypo': 'yellow',
            'Deep Hypo': 'red',
            'Unknown': 'gray'
        }
        
        # Filter palette to only include colors for categories actually present in the data
        actual_target_labels_in_data = df_plot[target_col_for_plot].cat.categories
        filtered_palette = {label: custom_palette_for_strings.get(label, 'lightgray') for label in actual_target_labels_in_data}


        if len(selected_features) == 2:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=selected_features[0], y=selected_features[1], hue=target_col_for_plot, data=df_plot, palette=filtered_palette, s=50, alpha=0.7, hue_order=df_plot[target_col_for_plot].cat.categories)
            plt.title(f'Scatter Plot di {selected_features[0]} vs {selected_features[1]} per {self.target_column} (Event)')
            plt.xlabel(selected_features[0])
            plt.ylabel(selected_features[1])
            plt.legend(title=f'{self.target_column} (Event)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()
        elif len(selected_features) >= 3:
            print("La visualizzazione di cluster con più di 2 feature è più complessa e richiederebbe tecniche di riduzione della dimensionalità (es. PCA, t-SNE) o librerie 3D.")
            g = sns.pairplot(df_plot, vars=selected_features[:3], hue=target_col_for_plot, palette=filtered_palette, diag_kind='kde', hue_order=df_plot[target_col_for_plot].cat.categories)
            g.fig.suptitle(f'Pair Plot delle Feature selezionate per {self.target_column} (Event)', y=1.02)
            plt.show()
        else:
            print("Non abbastanza feature selezionate per uno scatter plot (richiede almeno 2).")