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

class FeatureAnalyzer:
    """
    Classe per l'analisi delle feature estratte e analisi rispetto al target.
    """
    def __init__(self, data_file_path, target_column='target_hypoglycemia'):
        """
        Inizializza l'analizzatore di feature caricando il DataFrame.

        Args:
            data_file_path (str): Percorso del file CSV contenente le features estratte.
            target_column (str): Nome della colonna che contiene il target.
        """
        self.data_file_path = data_file_path
        self.target_column = target_column
        self.df = self._load_data()
        self.numeric_cols = []
        
        # Define target label mapping
        self.target_label_map = { # dictionary with the hypo classes
            0: 'No Hypo',
            1: 'Hypo',
            2: 'Deep Hypo'
        }

        if self.df is not None: 
            # Esclude colonne non numeriche o identificative dal calcolo delle feature
            # self.numeric_cols stores this list of numeric column names 
            self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist() # self.df.select_dtypes(include=np.number) ---> selects only those columns whose data type is considered a number by NumPy
                                                                                          # .columns.tolist() ---> extracts the names of these selected columns and converts them into a Python list

            if 'subj_id' in self.numeric_cols:
                self.numeric_cols.remove('subj_id')
                
            if self.target_column in self.numeric_cols:
                self.numeric_cols.remove(self.target_column) # Il target è un numero ma vogliamo analizzarlo separatamente

            # Assicurati che il target sia trattato come categoria
            if self.target_column in self.df.columns:
  
                # Create a new column with string labels for plotting
                # Map numeric values to string labels, fill unmapped (e.g., NaN) with 'Unknown'
                self.df[f'{self.target_column}_labeled'] = self.df[self.target_column].map(self.target_label_map).fillna('Unknown')
                
                # Ensure the order of categories for plotting
                # Get unique numeric values from original target (excluding NaN), map them to labels for ordering
                numeric_target_values = self.df[self.target_column].dropna().unique()
                # Create ordered_labels based on sorted numeric values that are in our map
                ordered_labels = [self.target_label_map[int(i)] for i in sorted(numeric_target_values) if int(i) in self.target_label_map]

                # Convert the labeled column to a Categorical type with explicit order
                self.df[f'{self.target_column}_labeled'] = pd.Categorical(self.df[f'{self.target_column}_labeled'], categories=ordered_labels, ordered=True)
                
                # The original target_column can remain as category of numbers if needed for other ops
                self.df[self.target_column] = self.df[self.target_column].astype('category')

            else:
                print(f"Warning: Colonna target '{target_column}' non trovata nel DataFrame.")
        
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
            "hypo_to_check_flag"]
        
        
        

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
        escludendo quelle specificate o utilizzando una lista specifica.
        """  
        return [f for f in self.numeric_cols if f not in self.features_to_exclude]
    

    def display_basic_info(self):
        """Mostra informazioni di base sul DataFrame."""
        if self.df is None: return
        print("\n--- Informazioni di Base sul DataFrame ---")
        print(self.df.info())
        # print("\n--- Statistiche Descrittive delle Feature Numeriche ---")
        # print(self.df[self.numeric_cols].describe())
        print(f"\n--- Distribuzione del Target ('{self.target_column}') ---")
        print(self.df[self.target_column].value_counts(dropna=False))
        
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
    def plot_focused_correlation(self, feature_list, title='Focused Correlation Heatmap', output_path=None, annotate_cells=True):
        """
        Generates a correlation heatmap for a specific subset of features.
        """
        valid_features = [f for f in feature_list if f in self.df.columns]
        if len(valid_features) < 2:
            print(f"Warning: Not enough valid features found in the DataFrame to plot '{title}'.")
            return

        correlation_matrix = self.df[valid_features].corr()

        plt.figure(figsize=(12, 10))
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
            

    def plot_distributions_by_target(self, num_cols_per_plot=3, figsize=(18, 5)):
        """
        Genera box plot per mostrare la distribuzione delle feature in 
        relazione al target.
        """
        if self.df is None or self.target_column not in self.df.columns or not self.numeric_cols:
            print("Nessun dato, colonna target o colonne numeriche disponibili per l'analisi per target.")
            return

        print(f"\n--- Plotting Distribuzioni delle Feature per Categoria di Target ({self.target_column}) ---")
        
        # Use the labeled target column for plotting
        target_col_for_plot = f'{self.target_column}_labeled'
        if target_col_for_plot not in self.df.columns:
            print(f"Errore: Colonna etichettata '{target_col_for_plot}' non trovata per il plotting.")
            return

        target_values = self.df[target_col_for_plot].dropna().unique()
        if len(target_values) > 10: # Evita troppe categorie
            print("Troppe categorie nel target per una visualizzazione efficace. Saltando.")
            return

        # Define custom palette for target labels using the string labels as keys
        custom_palette_for_strings = {
             'No Hypo': '#C8E6C9',  # verde tenue
             'Hypo': '#FFECB3',     # Giallo pastello
             'Deep Hypo': '#FFCDD2', # Rosso pastello
         }
        # Filter palette to only include colors for categories actually present in the data
        actual_target_labels_in_data = self.df[target_col_for_plot].cat.categories # Use .cat.categories for proper order
        
        filtered_palette = {label: custom_palette_for_strings.get(label, 'lightgray') for label in actual_target_labels_in_data}
        # 'lightgray' as a fallback for any labels not explicitly defined

        features_for_analysis = self._get_features_for_analysis()
            
        for i in range(0, len(features_for_analysis), num_cols_per_plot):
            cols_to_plot = features_for_analysis[i:i + num_cols_per_plot]
            fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(figsize[0], figsize[1]))
            if len(cols_to_plot) == 1:
                axes = [axes]

            for j, col in enumerate(cols_to_plot):
                
                # Only plot if the column contains at least one non-NaN value
                if self.df[col].notna().any():
                    sns.boxplot(x=target_col_for_plot, y=col, data=self.df, ax=axes[j], palette=filtered_palette, order=self.df[target_col_for_plot].cat.categories) # target_col_for_plot ---> 'No Hypo', 'Hypo', 'Deep Hypo'
                                                                                                                                                                     # y=col ---> specifies that the y-axis will represent the values of the current numerical feature (col)
                                                                                                                                                                     # order=self.df[target_col_for_plot].cat.categories: ---> explicitly sets the order of the categories on the x-axis
                
                    # A stripplot draws a scatter plot where one variable is categorical ---> often used in conjunction with box plots to show the individual data points (the raw observations) on top of the summarized distributions                    
                    sns.stripplot(x=target_col_for_plot, y=col, data=self.df, ax=axes[j], color='black', size=3, alpha=0.1, jitter=0.2, order=self.df[target_col_for_plot].cat.categories)  # color='black' ---> sets the color of the individual data points to black
                                                                                                                                                                                            # size=3 ---> sets the size of each data point (marker)
                                                                                                                                                                                            # alpha=0.1 ---> sets the transparency of the data points to 10%
                                                                                                                                                                                            # jitter=0.2 ---> adds a small amount of random noise to the positions of the points along the categorical (x) axis: this helps to prevent all points from stacking directly on top of each other, making it easier to visualize the density of points within each category
                    axes[j].set_title(f'{col}')
                    axes[j].set_title(f'{col}')
                    axes[j].set_xlabel('')
                else:
                    # If the column is empty, just display a message on the plot
                    axes[j].text(0.5, 0.5, 'No Data', ha='center', va='center')
                    axes[j].set_title(f'{col}')
                    axes[j].set_xlabel('')

                
                # Use the labeled target column and filtered palette
                #sns.boxplot(x=target_col_for_plot, y=col, data=self.df, ax=axes[j], palette=filtered_palette, order=self.df[target_col_for_plot].cat.categories) # target_col_for_plot ---> 'No Hypo', 'Hypo', 'Deep Hypo'
                                                                                                                                                                 # y=col ---> specifies that the y-axis will represent the values of the current numerical feature (col)
                                                                                                                                                                 # order=self.df[target_col_for_plot].cat.categories: ---> explicitly sets the order of the categories on the x-axis
            
                
                # A stripplot draws a scatter plot where one variable is categorical ---> often used in conjunction with box plots to show the individual data points (the raw observations) on top of the summarized distributions
                #sns.stripplot(x=target_col_for_plot, y=col, data=self.df, ax=axes[j], color='black', size=3, alpha=0.1, jitter=0.2, order=self.df[target_col_for_plot].cat.categories)  # color='black' ---> sets the color of the individual data points to black
                                                                                                                                                                                        # size=3 ---> sets the size of each data point (marker)
                                                                                                                                                                                        # alpha=0.1 ---> sets the transparency of the data points to 10%
                                                                                                                                                                                        # jitter=0.2 ---> adds a small amount of random noise to the positions of the points along the categorical (x) axis: this helps to prevent all points from stacking directly on top of each other, making it easier to visualize the density of points within each category
                #axes[j].set_title(f'{col}')
                #axes[j].set_xlabel('')
                
            plt.tight_layout()
            plt.show()


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