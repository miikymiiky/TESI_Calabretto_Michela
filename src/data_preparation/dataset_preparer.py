# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 10:46:33 2025

@author: prend
"""

# src/data_preparation/dataset_preparer.py

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec # For visualization helper

# Importa le funzioni di utilità generiche
from src.utils.utils4processing import (
    apply_zero_phase_low_pass_filter,
    calculate_derivative,
    re_position_meals,
    create_postprandial_dataframe
)

class DatasetPreparer:
    """
    Classe responsabile del caricamento, pre-processing dei dati CGM e wearable,
    riposizionamento dei pasti e creazione del database post-prandiale che 
    servirà poi per l'estrazione delle feature.
    """
    def __init__(self, data_dir="data", data_dir_db="postprandial_db",
                 glucose_filename_pattern="{}_tma.csv",
                 output_filename='all_postprandial_data.csv',
                 sampling_rate_seconds=300, cutoff_frequency_hz=2*1.67e-4, filter_order=1,
                 derivative_threshold=0.1, second_derivative_threshold=0.01,
                 search_window_before_meal=60, search_window_after_meal=30,
                 postprandial_duration_hours=3,
                 wearable_variable_names=['BATTERY_STATUS','HEART_RATE', 'HEART_RATE_VARIABILITY', 
                                          'PULSE_OX', 'RESPIRATION', 'STEPS', 'ZERO_CROSSING', 
                                          'ZERO_CROSSING_ENERGYTOTAL','ZERO_CROSSING_TIMEELAPSED']):
        """
        Inizializza il preparatore del dataset con i parametri necessari.

        Args:
            data_dir (str): Directory principale con i dati dei soggetti.
            data_dir_db (str): Directory dove salvare il database consolidato.
            glucose_filename_pattern (str): Pattern del nome file per il glucosio.
            sampling_rate_seconds (int): Frequenza di campionamento del glucosio in secondi.
            cutoff_frequency_hz (float): Frequenza di taglio del filtro passa-basso in Hz.
            filter_order (int): Ordine del filtro Butterworth.
            derivative_threshold (float): Soglia per la derivata prima per riposizionamento pasti.
            second_derivative_threshold (float): Soglia per la derivata seconda per riposizionamento pasti.
            search_window_before_meal (int): Minuti prima del pasto segnalato per la ricerca.
            search_window_after_meal (int): Minuti dopo il pasto segnalato per la ricerca.
            postprandial_duration_hours (int): Durata standard del periodo post-prandiale in ore.
            wearable_variable_names (list): Nomi dei file/variabili dei dati wearable.
        """
        self.data_dir = data_dir
        self.data_dir_db = data_dir_db
        self.glucose_filename_pattern = glucose_filename_pattern
        self.sampling_rate_seconds = sampling_rate_seconds
        self.cutoff_frequency_hz = cutoff_frequency_hz
        self.filter_order = filter_order
        self.derivative_threshold = derivative_threshold
        self.second_derivative_threshold = second_derivative_threshold
        self.search_window_before_meal = search_window_before_meal
        self.search_window_after_meal = search_window_after_meal
        self.postprandial_duration_hours = postprandial_duration_hours
        self.wearable_variable_names = wearable_variable_names
        self.output_filename=output_filename

        # Crea la directory del database se non esiste
        os.makedirs(self.data_dir_db, exist_ok=True)
    
    def _load_glucose_dataframe(self, subject_id):
        """Carica il DataFrame del glucosio per un soggetto specifico."""
        glucose_df_path = os.path.join(self.data_dir, subject_id, self.glucose_filename_pattern.format(subject_id)) # Carica il file originale non processato
                                                                                                                    # .format(subject_id) replaces the {} placeholder with the actual subject_id
        try:
            glucose_df = pd.read_csv(glucose_df_path, index_col='Time', parse_dates=True) # index_col='Time' ---> specifies that the column named 'Time' in the CSV should be used as the DataFrame's index
                                                                                          # parse_dates=True ---> tells pandas to automatically parse the values in the index column as datetime objects
            return glucose_df
        except FileNotFoundError:
            print(f"Warning: File glucosio non trovato per il soggetto {subject_id} a {glucose_df_path}")
            return None
        except Exception as e:
            print(f"Errore durante il caricamento del glucosio per {subject_id}: {e}")
            return None

    def _load_wearable_dataframes(self, subject_id):
        """Carica i DataFrames wearable per un soggetto specifico."""
        all_wearable_data = {} # initializes an empty dictionary
        for var_name in self.wearable_variable_names:
            wearable_df_path = os.path.join(self.data_dir, subject_id, f"{var_name}.csv")
            if os.path.exists(wearable_df_path):
                try:
                    df = pd.read_csv(wearable_df_path, parse_dates=['Time']).set_index('Time') # .set_index('Time') ---> sets the 'Time' column as the DataFrame's index
                    all_wearable_data[var_name] = df
                except Exception as e:
                    print(f"Warning: Errore caricamento {var_name}.csv")
            else:
                print(f"Warning: File {var_name}.csv non trovato")
        return all_wearable_data

    def _plot_subject_data(self, subject_df, subject_id, all_wearable_data):
        """
        Visualizza il glucosio, le sue derivate e i dati wearable per un soggetto,
        evidenziando i pasti segnalati e riposizionati.
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.4) # Spazio verticale tra i subplots
                                                                             # 4, 1 ---> grid of 4 rows and 1 column
                                                                             # height_ratios=[3, 1, 1, 1] ---> allocates relative heights to the rows: the first row (for glucose and derivatives) will be 3 times taller than the other three rows (for wearable data)
                                                                             # hspace=0.4 ---> sets the height spacing between subplots to 0.4 of the average height of a subplot

        # Glucosio e Derivate
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(subject_df.index, subject_df['glucose'], label='CGM Originale', color='lightgray', alpha=0.7)
        ax0.plot(subject_df.index, subject_df['glucose_filtered'], label='CGM Filtrato', color='blue')
        ax0.plot(subject_df.index, subject_df['glucose_derivative'], label='Derivata 1°', color='green', linestyle='--')
        ax0.plot(subject_df.index, subject_df['glucose_second_derivative'], label='Derivata 2°', color='purple', linestyle=':')

        # Markers per i pasti
        meal_reported_times = subject_df[subject_df['meal'] == 1].index
        meal_tma_times = subject_df[subject_df['TMA_meal'] == 1].index
        meal_tma_corrected_times = subject_df[subject_df['TMA_meal_corrected'] == 1].index
        
        # Aggiusta la y per non sovrapporsi ai picchi di glucosio
        y_val_offset = subject_df['glucose_filtered'].max() * 0.05 if not subject_df['glucose_filtered'].empty else 0
        
        ax0.scatter(meal_reported_times, subject_df.loc[meal_reported_times, 'glucose_filtered'] + y_val_offset, 
                    marker='^', color='red', s=100, label='Pasto Segnalato (Originale)')
        
        ax0.scatter(meal_tma_times, subject_df.loc[meal_tma_times, 'glucose_filtered'] + y_val_offset * 2, # s=100 ---> sets the size of the markers
                    marker='x', color='orange', s=100, label='Pasto (TMA)')
        
        ax0.scatter(meal_tma_corrected_times, subject_df.loc[meal_tma_corrected_times, 'glucose_filtered'] + y_val_offset * 2, 
                    marker='o', color='orange', s=100, label='Pasto Riposizionato (TMA)')
        
        ax0.set_ylabel('Glucosio [mg/dL] / Derivate')
        ax0.set_title(f'Soggetto: {subject_id} - Glucosio e Derivate con Riposizionamento Pasti')
        ax0.legend(loc='upper left')
        ax0.grid(True)

        # Plot dei dati Wearable
        ax_wearable_start_idx = 1

        for i, var_name in enumerate(['HEART_RATE', 'STEPS', 'PULSE_OX']): # Plotta solo alcuni per chiarezza
            if var_name in all_wearable_data and not all_wearable_data[var_name].empty:
                
                wearable_index_time = pd.to_datetime(all_wearable_data[var_name].index, format='%d-%b-%Y %H:%M:%S')
                wearable_values = all_wearable_data[var_name][var_name]
                
                ax = fig.add_subplot(gs[ax_wearable_start_idx + i], sharex=ax0)
                ax.plot(wearable_index_time, wearable_values, label=var_name, color='teal')
                
                # Evidenzia i pasti riposizionati anche sui grafici wearable
                for meal_time in meal_tma_corrected_times:
                    meal_end = meal_time + pd.Timedelta(hours=self.postprandial_duration_hours)
                    ax.axvspan(meal_time, meal_end, color='orange', alpha=0.1) # ax.axvspan() ---> draws a vertical shaded span on the wearable plot, highlighting the 3-hour post-prandial period associated with each corrected meal

                ax.set_ylabel(var_name)
                ax.legend(loc='upper left')
                ax.grid(True)
            else:
                pass # L'asse verrà creato ma rimarrà vuoto

        plt.xlabel('Tempo')
        plt.tight_layout()
        plt.show()

    def prepare_dataset(self, visualize_each_subject=False):
        """
        Esegue l'intero processo di preparazione del dataset per tutti i soggetti.

        Args:
            visualize_each_subject (bool): Se True, visualizza i grafici per ogni soggetto dopo il processing.
        """
        all_subject_folders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        
        all_postprandial_data = [] # initializes an empty list that will collect the postprandial_df_subj

        print("Inizio preparazione del dataset...")
        for i, subject_folder in enumerate(all_subject_folders):
            
            subject_id = subject_folder # nome cartella coincide con ID del soggetto
            
            
            # EXCLUDE SUBJECT 25 from the analysis
            if subject_id == 'BERNPBHF00025': 
                print(f"\nSkipping excluded subject: {subject_id}")
                continue
            
            print(f"\nProcessing Subject: {subject_id} ({i+1}/{len(all_subject_folders)})")

            # 1. Carica i dati
            subject_df = self._load_glucose_dataframe(subject_id)
            if subject_df is None or subject_df.empty:
                print(f"Skipping {subject_id} due to missing or empty glucose data.")
                continue

            all_wearable_data = self._load_wearable_dataframes(subject_id)

            # Esegui il filtro solo se ci sono dati non NaN
            if 'glucose' in subject_df.columns and not subject_df['glucose'].dropna().empty:
                # Applica filtro passa-basso
                subject_df['glucose_filtered'] = apply_zero_phase_low_pass_filter(
                    subject_df['glucose'].values,
                    self.sampling_rate_seconds,
                    self.cutoff_frequency_hz,
                    self.filter_order
                )
            else:
                subject_df['glucose_filtered'] = np.nan
                print(f"Attenzione: Colonna 'glucose' vuota o con soli NaN per {subject_id}. Non applico il filtro.")


            # Calcola derivate (anche su valori NaN, verranno ripristinati)
            # Passa l'indice come tempo per calculate_derivative
            subject_df['glucose_derivative'], subject_df['glucose_second_derivative'] = \
                calculate_derivative(subject_df['glucose_filtered'].values, subject_df.index)

            # Riposiziona i pasti
            # Passa sampling_rate_minutes a re_position_meals
            subject_df['TMA_meal_corrected'] = re_position_meals(
                subject_df.copy(), # Passa una copia per evitare SettingWithCopyWarning
                self.derivative_threshold, self.second_derivative_threshold,
                self.search_window_before_meal, self.search_window_after_meal,
                self.sampling_rate_seconds / 60 # Sampling rate in minuti
            )
            
            # Visualizza se richiesto
            if visualize_each_subject:
                self._plot_subject_data(subject_df, subject_id, all_wearable_data)

            # Crea dataframe per i pasti post-prandiali del soggetto
            postprandial_df_subj = create_postprandial_dataframe(
                subject_df, all_wearable_data, subject_id, self.postprandial_duration_hours
            )
            
            all_postprandial_data.append(postprandial_df_subj)

            # Salva il file processato del soggetto
            output_path_subject = os.path.join(self.data_dir, subject_folder, subject_folder + 'tma_processed_new.csv')
            subject_df.to_csv(output_path_subject)
            print(f"Salvato dati processati del soggetto a: {output_path_subject}")
        
        # Concatena tutti i DataFrame post-prandiali
        if all_postprandial_data:
            final_postprandial_df = pd.concat(all_postprandial_data, ignore_index=True)
            output_path_db = os.path.join(self.data_dir_db, self.output_filename)
            final_postprandial_df.to_csv(output_path_db, index=False)
            print(f"\n-->Preparazione dataset completata. Database salvato a: {output_path_db}<--")
            return final_postprandial_df
        else:
            print("\nNessun dato post-prandiale elaborato.")
            return pd.DataFrame()

# Esempio di utilizzo (per testare solo questa classe)
if __name__ == "__main__":
    preparer = DatasetPreparer(
        data_dir="data",
        data_dir_db="postprandial_db",
        sampling_rate_seconds=300,
        cutoff_frequency_hz=2*1.67e-4,
        filter_order=1,
        derivative_threshold=0.1,
        second_derivative_threshold=0.01,
        search_window_before_meal=60,
        search_window_after_meal=30,
        postprandial_duration_hours=3
    )
    # Esegui la preparazione del dataset, visualizzando i grafici per ogni soggetto
    # Assicurati di avere dati nella cartella 'data' strutturata come 'data/subj_001/glucose.csv' ecc.
    # Se imposti visualize_each_subject=True, si apriranno molti plot.
    final_df = preparer.prepare_dataset(visualize_each_subject=False)
    print(final_df.head())