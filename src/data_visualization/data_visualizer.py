# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:31:06 2025

@author: prend
"""

# src/visualization/data_visualizer.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np # Importato per np.nan e np.isna

class DataVisualizer:
    """
    Classe per la visualizzazione dei dati CGM e wearable.
    """
    def __init__(self, data_dir="data", data_dir_db="postprandial_db",
                 postprandial_data_file='all_postprandial_data_with_features.csv', # Aggiornato per usare il file con le feature
                 glucose_filename_pattern="{}tma_processed_new.csv", #"{}_tma_processed.csv",
                 heart_rate_filename="HEART_RATE.csv",
                 steps_filename="STEPS.csv",
                 pulse_ox_filename="PULSE_OX.csv"):
        """
        Inizializza il visualizzatore con i percorsi dei dati e i nomi dei file.

        Args:
            data_dir (str): Directory principale con i dati dei soggetti.
            data_dir_db (str): Directory del database post-prandiale (all_postprandial_data.csv).
            postprandial_data_file (str): Nome del file del database post-prandiale (con feature).
            glucose_filename_pattern (str): Pattern del nome file per i dati di glucosio processati.
            heart_rate_filename (str): Nome del file per i dati di frequenza cardiaca.
            steps_filename (str): Nome del file per i dati dei passi.
            pulse_ox_filename (str): Nome del file per i dati di saturazione dell'ossigeno.
        """
        self.data_dir = data_dir
        self.data_dir_db = data_dir_db
        self.postprandial_data_file = postprandial_data_file
        self.glucose_filename_pattern = glucose_filename_pattern
        self.heart_rate_filename = heart_rate_filename
        self.steps_filename = steps_filename
        self.pulse_ox_filename = pulse_ox_filename
        
        self.all_postprandial_df = self._load_all_postprandial_data()

    def _load_all_postprandial_data(self):
        """Carica il DataFrame principale con tutti i dati post-prandiali (inclusi feature)."""
        file_path = os.path.join(self.data_dir_db, self.postprandial_data_file)
        try:
            df = pd.read_csv(file_path)
            df['meal_time_start'] = pd.to_datetime(df['meal_time_start'], format='%Y-%m-%d %H:%M:%S')
            df['meal_time_end'] = pd.to_datetime(df['meal_time_end'], format='%Y-%m-%d %H:%M:%S')
            print(f"Caricato {self.postprandial_data_file} con successo.")
            return df
        except FileNotFoundError:
            print(f"Errore: File dati post-prandiali non trovato a {file_path}. Assicurati che esista e sia stato generato dalla classe PostprandialFeatureExtractor.")
            return pd.DataFrame() # Ritorna un DataFrame vuoto
        except Exception as e:
            print(f"Si è verificato un errore durante il caricamento dei dati post-prandiali: {e}")
            return pd.DataFrame()

    def _load_subject_specific_data(self, subject_id):
        """Carica i DataFrames specifici per un soggetto (glucosio e wearable)."""
        glucose_df_path = os.path.join(self.data_dir, subject_id, self.glucose_filename_pattern.format(subject_id))
        glucose_df = None
        try:
            glucose_df = pd.read_csv(glucose_df_path, index_col='Time', parse_dates=True)
        except FileNotFoundError:
            print(f"Errore: File glucosio non trovato per il soggetto {subject_id} a {glucose_df_path}")
            return None, None, None, None
        except Exception as e:
            print(f"Errore caricamento glucosio per {subject_id}: {e}")
            return None, None, None, None

        heart_rate_df = self._load_wearable_file(subject_id, self.heart_rate_filename)
        steps_df = self._load_wearable_file(subject_id, self.steps_filename)
        pulse_ox_df = self._load_wearable_file(subject_id, self.pulse_ox_filename)

        return glucose_df, heart_rate_df, steps_df, pulse_ox_df

    def _load_wearable_file(self, subject_id, filename):
        """Helper per caricare un singolo file wearable."""
        file_path = os.path.join(self.data_dir, subject_id, filename)
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path, parse_dates=['Time']).set_index('Time')
            except Exception as e:
                print(f"Attenzione: Errore caricamento {filename} per il soggetto {subject_id}: {e}")
                return None
        else:
            # print(f"Attenzione: File {filename} non trovato per il soggetto {subject_id}") # Commentato per ridurre verbosità
            return None

    def plot_meal_context(self, subject_id, meal_time_start_selected):
        """
        Visualizza la traccia completa del soggetto con l'intervallo postprandiale evidenziato.

        Args:
            subject_id (str): ID del soggetto.
            meal_time_start_selected (pd.Timestamp): Ora di inizio del pasto da evidenziare.
        """
        if self.all_postprandial_df.empty:
            print("Impossibile tracciare: Dati post-prandiali non caricati.")
            return

        meal_row_candidates = self.all_postprandial_df[
            (self.all_postprandial_df['subj_id'] == subject_id) &
            (self.all_postprandial_df['meal_time_start'] == meal_time_start_selected)
        ]
        
        if meal_row_candidates.empty:
            print(f"Nessun pasto trovato per il soggetto {subject_id} all'ora {meal_time_start_selected}.")
            return
        
        meal_row = meal_row_candidates.iloc[0] # Prendi la prima corrispondenza

        meal_start = meal_row['meal_time_start']
        meal_end = meal_row['meal_time_end']

        glucose_df, heart_rate_df, steps_df, pulse_ox_df = self._load_subject_specific_data(subject_id)

        if glucose_df is None:
            print(f"Impossibile caricare i dati glucosio per {subject_id}, impossibile tracciare.")
            return

        # Prepara i dati per il plotting basandosi sui nomi delle colonne attesi
        meal_reported_times = glucose_df[glucose_df['meal'] == 1].index if 'meal' in glucose_df.columns else pd.DatetimeIndex([])
        meal_template_times = glucose_df[glucose_df['TMA_meal'] == 1].index if 'TMA_meal' in glucose_df.columns else pd.DatetimeIndex([])
        meal_corrected_times = glucose_df[glucose_df['TMA_meal_corrected'] == 1].index if 'TMA_meal_corrected' in glucose_df.columns else pd.DatetimeIndex([])

        # Estrai le feature dal meal_row
        time_to_peak_min = meal_row.get('time_to_first_glucose_peak_after_meal')
        peak_value = meal_row.get('first_glucose_peak_value_after_meal')
        second_peak_value = meal_row.get('second_glucose_peak_value_after_meal')
        time_to_second_peak_min = meal_row.get('time_to_second_glucose_peak_after_meal')
        hypo_time_stamp = meal_row.get('postprandial_hypo_time') # Questo è già un Timestamp se caricato correttamente
        deep_hypo_time_stamp = meal_row.get('postprandial_deep_hypo_time') # Questo è già un Timestamp
        is_hypo = meal_row.get('postprandial_hypo', 0)

        # Converti i Timestamp di ipoglicemia in Datetime se sono stringhe o simili
        if isinstance(hypo_time_stamp, str):
            hypo_time_stamp = pd.to_datetime(hypo_time_stamp)
        if isinstance(deep_hypo_time_stamp, str):
            deep_hypo_time_stamp = pd.to_datetime(deep_hypo_time_stamp)


        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        # --- Subplot 1: Glucosio con fasce colorate e markers ---
        # Fasce di glucosio
        axs[0].axhspan(0, 54, color='red', alpha=0.2, label='Glucosio < 54 mg/dL (Ipoglicemia Profonda)')
        axs[0].axhspan(54, 70, color='yellow', alpha=0.2, label='Glucosio < 70 mg/dL (Ipoglicemia)')
        axs[0].axhspan(70, 180, color='green', alpha=0.2, label='Glucosio 70-180 mg/dL (Range Normale)')

        axs[0].plot(glucose_df.index, glucose_df['glucose'], color='lightgray', alpha=0.7, label='CGM Originale')
        # Controlla se la colonna glucose_filtered esiste prima di plottarla
        if 'glucose_filtered' in glucose_df.columns:
            axs[0].plot(glucose_df.index, glucose_df['glucose_filtered'], color='blue', label='CGM Filtrato')
            axs[0].plot(glucose_df.loc[meal_start:meal_end].index, glucose_df.loc[meal_start:meal_end]['glucose_filtered'], color='red', label='CGM Filtrato Post-prandiale')
            y_offset_col = 'glucose_filtered' # Usa questa colonna per l'offset dei marker
        else:
            axs[0].plot(glucose_df.loc[meal_start:meal_end].index, glucose_df.loc[meal_start:meal_end]['glucose'], color='red', label='CGM Originale Post-prandiale')
            y_offset_col = 'glucose' # Fallback all'originale

        # Markers Pasti
        if not meal_reported_times.empty:
            axs[0].scatter(meal_reported_times, glucose_df.loc[meal_reported_times][y_offset_col] + 10, marker='^', color='red', label='Pasto Segnalato', s=100)
        if not meal_template_times.empty:
            axs[0].scatter(meal_template_times, glucose_df.loc[meal_template_times][y_offset_col] + 20, marker='x', color='orange', label='Pasto Template', s=100)
        if not meal_corrected_times.empty:
            axs[0].scatter(meal_corrected_times, glucose_df.loc[meal_corrected_times][y_offset_col] - 10, marker='o', color='green', label='Pasto Corretto', s=100)


        # Markers Picchi
        if not pd.isna(peak_value) and not pd.isna(time_to_peak_min):
            peak_time = meal_start + pd.Timedelta(minutes=time_to_peak_min)
            axs[0].plot(peak_time, peak_value, marker='^', color='purple', markersize=10, label=f'Picco 1 ({peak_value:.1f} mg/dL)')

        if not pd.isna(second_peak_value) and not pd.isna(time_to_second_peak_min):
            second_peak_time = meal_start + pd.Timedelta(minutes=time_to_second_peak_min)
            axs[0].plot(second_peak_time, second_peak_value, marker='v', color='brown', markersize=10, label=f'Picco 2 ({second_peak_value:.1f} mg/dL)')
            
        # Linee Ipoglicemia (usa i timestamp diretti)
        if pd.notna(hypo_time_stamp):
            axs[0].axvline(hypo_time_stamp, color='red', linestyle='--', label='Inizio Ipoglicemia (<70)')
        if pd.notna(deep_hypo_time_stamp):
            axs[0].axvline(deep_hypo_time_stamp, color='darkred', linestyle=':', label='Inizio Ipoglicemia Profonda (<54)')


        axs[0].set_ylabel('Glucosio [mg/dL]')
        axs[0].legend(loc='upper right')

        # --- Subplot 2: Frequenza Cardiaca ---
        if heart_rate_df is not None:
            heart_rate_df.index = pd.to_datetime(heart_rate_df.index, format='%d-%b-%Y %H:%M:%S')
            
            axs[1].plot(heart_rate_df.index, heart_rate_df['HEART_RATE'], color='lightgray', alpha=0.7, label='Frequenza Cardiaca')
            axs[1].plot(heart_rate_df.loc[meal_start:meal_end].index, heart_rate_df.loc[meal_start:meal_end]['HEART_RATE'], color='red', label='FC Post-prandiale')
            axs[1].set_ylabel('Frequenza Cardiaca')
            axs[1].legend()
        else:
            axs[1].set_visible(False) # Nascondi l'asse se i dati non sono disponibili

        # --- Subplot 3: Passi ---
        if steps_df is not None:
            steps_df.index = pd.to_datetime(steps_df.index, format = '%d-%b-%Y %H:%M:%S')
            axs[2].plot(steps_df.index, steps_df['STEPS'], color='lightgray', alpha=0.7, label='Passi')
            axs[2].plot(steps_df.loc[meal_start:meal_end].index, steps_df.loc[meal_start:meal_end]['STEPS'], color='red', label='Passi Post-prandiale')
            axs[2].set_ylabel('Passi')
            axs[2].legend()
        else:
            axs[2].set_visible(False)

        # --- Subplot 4: Saturazione Ossigeno ---
        if pulse_ox_df is not None:
            pulse_ox_df.index = pd.to_datetime(pulse_ox_df.index, format = '%d-%b-%Y %H:%M:%S')
            axs[3].plot(pulse_ox_df.index, pulse_ox_df['PULSE_OX'], color='lightgray', alpha=0.7, label='Saturazione Ossigeno (%)')
            axs[3].plot(pulse_ox_df.loc[meal_start:meal_end].index, pulse_ox_df.loc[meal_start:meal_end]['PULSE_OX'], color='red', label='Saturazione Ossigeno Post-prandiale')
            axs[3].set_ylabel('Saturazione Ossigeno (%)')
            axs[3].legend()
        else:
            axs[3].set_visible(False)

        axs[3].set_xlabel('Tempo')
        plt.suptitle(f"Soggetto: {subject_id}, Inizio Pasto: {meal_start.strftime('%Y-%m-%d %H:%M')}" + (" (Ipoglicemia)" if is_hypo else ""), fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Regola il layout per evitare sovrapposizioni del titolo
        plt.show()

    def run_interactive_visualizer(self):
        """
        Avvia l'interfaccia interattiva a riga di comando per selezionare e visualizzare i pasti.
        Permette all'utente di scegliere un soggetto, poi un pasto, e visualizzarlo.
        L'utente può continuare a visualizzare pasti, tornare alla selezione del soggetto o uscire.
        """
        if self.all_postprandial_df.empty:
            print("Impossibile avviare il visualizzatore interattivo: Dati post-prandiali non caricati.")
            return

        while True:
            subjects = self.all_postprandial_df['subj_id'].unique()
            subjects.sort() # Ordina per visualizzazione consistente

            print("\nSoggetti Disponibili:")
            for i, subj in enumerate(subjects):
                print(f"{i + 1}. {subj}")

            selected_subject = None
            while selected_subject is None:
                subject_input = input("Inserisci il numero del soggetto da visualizzare (o 'q' per uscire): ").strip()
                if subject_input.lower() == 'q':
                    print("Uscita dal visualizzatore.")
                    return # Esci completamente
                
                try:
                    subject_index = int(subject_input) - 1
                    if 0 <= subject_index < len(subjects):
                        selected_subject = subjects[subject_index]
                    else:
                        print("Selezione soggetto non valida. Inserisci un numero valido.")
                except ValueError:
                    print("Input non valido. Inserisci un numero.")

            subject_postprandial_df = self.all_postprandial_df[
                self.all_postprandial_df['subj_id'] == selected_subject
            ].copy()
            subject_postprandial_df = subject_postprandial_df.sort_values(by='meal_time_start').reset_index(drop=True)

            if subject_postprandial_df.empty:
                print(f"Nessun pasto trovato per il soggetto {selected_subject}.")
                continue # Torna alla selezione del soggetto

            print(f"\nPasti Disponibili per il Soggetto {selected_subject}:")
            for i, row in subject_postprandial_df.iterrows():
                meal_time_start = row['meal_time_start']
                # Accedi alle colonne con .get per sicurezza
                hypo_status = row.get('postprandial_hypo', 'N/A')
                deep_hypo_status = row.get('postprandial_deep_hypo', 'N/A')
                print(f"{i + 1}. Inizio Pasto: {meal_time_start.strftime('%Y-%m-%d %H:%M')} (Hypo: {hypo_status}, Deep Hypo: {deep_hypo_status})")

            while True:
                meal_input = input("Inserisci il numero del pasto da visualizzare (o 'q' per uscire, 'b' per tornare ai soggetti): ").strip()
                if meal_input.lower() == 'q':
                    print("Uscita dal visualizzatore.")
                    return # Esci completamente
                if meal_input.lower() == 'b':
                    break # Esci dal ciclo di selezione pasto e torna alla selezione soggetto
                
                try:
                    meal_index = int(meal_input) - 1
                    if 0 <= meal_index < len(subject_postprandial_df):
                        selected_meal_time_start = subject_postprandial_df.iloc[meal_index]['meal_time_start']
                        self.plot_meal_context(selected_subject, selected_meal_time_start)
                    else:
                        print("Selezione pasto non valida. Inserisci un numero valido.")
                except ValueError:
                    print("Input non valido. Inserisci un numero o 'q'/'b'.")
                except Exception as e:
                    print(f"Si è verificato un errore durante la visualizzazione: {e}")
                    # Continua il loop in caso di errore per permettere all'utente di riprovare


# Esempio di utilizzo (per testare solo questa classe)
if __name__ == "__main__":
    # Assicurati che 'all_postprandial_data_with_features.csv' sia stato generato da PostprandialFeatureExtractor
    visualizer = DataVisualizer(
        data_dir="data",
        data_dir_db="postprandial_db",
        postprandial_data_file='all_postprandial_data_with_features.csv'
    )
    
    # Avvia il visualizzatore interattivo
    visualizer.run_interactive_visualizer()