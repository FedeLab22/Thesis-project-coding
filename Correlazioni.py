# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:28:34 2025

@author: fefe2
"""

from merged import merged_file
from merged import get_parametri
from merged import create_xls_merge
import copy
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

c = 2759

class CorrelationAndPlotter:
    def __init__(self, data_dict):
        # Stessa inizializzazione come prima
        self.data_dict = data_dict
        self.input_columns = ['ID', 'Volume', 'Vel trasl blocco', 'Angolo impatto', 'Pos impatto Z']
        self.output_columns = ['U1_T_Block', 'V1_T_Block', 'U3_T_Block', 'U2_T_Block', 'KE_Block', 'V3_T_Block', 'V2_T_Block', 'FRECCIA_MAX_CAVOSUP', 'A_T_Block', 'V_T_Block', 'V_R_Block', 'RF_MDX', 'RF_MSX', 'RM_MDX', 'RM_MSX', 'E_TRASL', 'E_ROT', 'A_TOT']
        self.output_03 = ['U1_T_Block', 'U2_T_Block', 'U3_T_Block', 'U1_CL', 'U3_CL', 'DeltaDT', 'DeltaDT%', 'KE_Block', 'E_TRASL', 'E_ROT']
        self.output_atot_max = ['V1_T_Block', 'V2_T_Block', 'V3_T_Block', 'V_T_Block', 'V_R_Block', 'A_TOT', 'FRECCIA_MAX_CAVOSUP', 'RF_MDX', 'RF_MSX', 'RM_MDX', 'RM_MSX']
        self.list_excludes = ['Status', 'tempo_units', 'RM1_MSX_units', 'RF3_MDX_units', 'U3_CAVOSUP_units', 'U2_T_Block_units', 'KE_Block_units', 'A3_T_Block_units', 'A1_T_Block_units', 'V3_T_Block_units', 'RF1_MDX_units', 'U2_CAVOSUP_units', 'RM3_MDX_units', 'RF3_MSX_units', 'RF2_MSX_units', 'RF1_MSX_units', 'V1_R_Block_units', 'U3_T_Block_units', 'U1_CAVOSUP_units', 'RM2_MSX_units', 'RM3_MSX_units', 'V3_R_Block_units', 'A2_T_Block_units', 'V2_R_Block_units', 'V1_T_Block_units', 'RM1_MDX_units', 'RF2_MDX_units', 'V2_T_Block_units', 'U1_T_Block_units', 'RM2_MDX_units', '']
        self.data_filled = None
        self.STATUS_FAILURE = "Failure"
        self.max_length = 0
        all_data = []
        
        # Trova la lunghezza massima delle liste
        for key, value in self.data_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.max_length = max(self.max_length, len(sub_value))
        
        indices_to_remove = []
        # Normalizza le lunghezze delle liste
        for key, value in self.data_dict.items():
            if isinstance(value, dict):
                if not indices_to_remove:
                    indices_to_remove = [i for i, v in enumerate(value["Status"]) if v == self.STATUS_FAILURE]
                for sub_key, sub_value in value.items():
                    if len(sub_value) < self.max_length and type(sub_value) == type([]):
                        value[sub_key] = sub_value + [np.nan] * (self.max_length - len(sub_value))
                for sub_key, sub_value in value.copy().items():
                    value[sub_key] = [v for i, v in enumerate(sub_value) if i not in indices_to_remove]
                    if sub_key in self.list_excludes:
                        value.pop(sub_key)
                all_data.append(pd.DataFrame(value))
        
        # Combina i dati in un unico DataFrame
        self.data_filled = pd.concat(all_data)

    def calculate_and_plot_correlations(self, type=None, show=True):
        df = self.data_filled  # Usa il DataFrame già popolato
        
        if 'ID' not in df.columns or 'A_TOT' not in df.columns or 'tempo' not in df.columns:
            print("Colonne 'ID' o 'A_TOT' o 'Tempo' non trovate!")
            return

        # Calcolare la traiettoria senza intercettazione (U1_CL, U3_CL)
        g = 9.81  # Accelerazione di gravità
        teta = 60
        teta_rad = np.radians(teta) 
        df['U1_CL'] = np.nan
        df['U3_CL'] = np.nan

        # Aggiorna il dataframe originale con le nuove colonne
        for id_value, group in df.groupby('ID'):
            # Estrai velocità iniziale per ogni gruppo (ID)
            V1_T_block_0 = group.sort_values(by="tempo").iloc[0]['V1_T_Block']
            V3_T_block_0 = group.sort_values(by="tempo").iloc[0]['V3_T_Block']
            #Pos_impatto_Z = group['Pos impatto Z'].values[0]  # U3_in lo sarebbe se il sistema di riferimento fosse lo stesso per il pendio e il blocco
            #il sistema di riferimento del blocco è il blocco stesso 

            # Calcola le traiettorie per ogni tempo (t) nel gruppo
            for i, row in group.iterrows():
                t = row['tempo']
                U1_in = 0  # U1_in sempre 0
                #U2_in = 0
                U3_in = 0 #Pos_impatto_Z  # Posizione di impatto Z per ogni ID
                
                # Calcola le nuove posizioni (U1_CL, U3_CL) basate sulla formula della traiettoria
                U1_CL = U1_in + V1_T_block_0 * np.cos(teta_rad) * t + 0.5 * g * np.sin(teta_rad) * np.cos(teta_rad) * t**2
                U3_CL = U3_in + V3_T_block_0 * np.sin(teta_rad) * t - 0.5 * g * np.sin(teta_rad) * np.sin(teta_rad) * t**2
                
                # Aggiorna direttamente le colonne del DataFrame originale
                df.at[i, 'U1_CL'] = U1_CL
                df.at[i, 'U3_CL'] = U3_CL

        # Calcolare DeltaDT e DeltaDT% per ogni ID nel DataFrame originale
        if 'U1_T_Block' in df.columns and 'U1_CL' in df.columns:
            df['DeltaDT'] =  df['U1_CL'] - df['U1_T_Block'] 
            df['DeltaDT%'] = ( df['U1_CL'] - df['U1_T_Block']) / df['U1_CL'] * 100
        else:
            print("Le colonne U1_T_Block o U1_CL non sono presenti nel DataFrame!")
        
        # Prendi la prima riga di input per ogni ID
        first_rows_input = df.loc[df.groupby('ID').head(1).index, self.input_columns]
        #print ("Prima colonna di input per id:", first_rows_input)
        
        result_data = []

        # Raggruppa per ID
        grouped = self.data_filled.groupby('ID')

        for id_value, group in grouped:
            # Trova la riga con t=0 (input) e la riga con t=0.3 (output)
            input_row = group.sort_values(by="tempo").iloc[0]  # Prendi la riga per t=0
            output_row = group.sort_values(by="tempo").iloc[-1]
            
            # Estrai solo le colonne di output desiderate
            output_values = output_row[self.output_03].to_dict()
            
            # Estrai solo le colonne di input desiderate
            input_values = input_row[self.input_columns].to_dict()

            # Combina i dati di input e output
            combined_row = {**input_values, **output_values}

            result_data.append(combined_row)

        # Crea un DataFrame con il risultato
        result_df = pd.DataFrame(result_data)
        print(result_df[self.output_03].dtypes)
        # Ora, per la parte di A_TOT massimo, facciamo la stessa cosa senza sovrascrivere i dati

        relevant_rows_atot_max = []  # Lista per contenere le righe intorno al massimo di A_TOT
        input_rows_repeated_atot_max = []  # Lista per contenere le righe di input ripetute

        # 4. Estrai le 5 righe intorno al massimo valore di A_TOT
        for id_value, group in df.groupby("ID"):
            max_index = group["A_TOT"].idxmax()  # Trova l'indice del massimo A_TOT
            selected_rows = group.iloc[max(max_index - 2, 0): min(max_index + 3, len(group))]  # Prendi le righe attorno al massimo
            relevant_rows_atot_max.append(selected_rows[['ID'] + self.output_atot_max])  # Prendi solo gli output del gruppo a_tot_max

        # 5. Concatenare tutte le righe di relevant_rows per A_TOT massimo
        relevant_rows_df_atot_max = pd.concat(relevant_rows_atot_max, ignore_index=True)

        # Aggiungi la prima riga di input per ciascun ID per A_TOT massimo
        for id_value, group in relevant_rows_df_atot_max.groupby("ID"):
            if id_value in first_rows_input['ID'].values:
                # Trova la prima riga di input corrispondente all'ID (tempo=0)
                input_row_atot_max = first_rows_input.loc[first_rows_input['ID'] == id_value].iloc[0]

                # Ripeti la riga di input per il numero di righe di output (5 righe per ogni ID)
                input_rows_repeated_atot_max.extend([input_row_atot_max] * len(group))  # Aggiungi input ripetuti

        # 6. Creare il DataFrame con le righe di input ripetute per A_TOT massimo
        input_rows_df_atot_max = pd.DataFrame(input_rows_repeated_atot_max)
        
        # 7. Unire gli output a A_TOT massimo con gli input ripetuti
        merged_df_atot_max = pd.concat([relevant_rows_df_atot_max.reset_index(drop=True), input_rows_df_atot_max.reset_index(drop=True)], axis=1)

        # Stampa i nomi delle colonne per A_TOT massimo
        #print("Columns for A_TOT max:", merged_df_atot_max.columns)

        # Calcola la matrice di correlazione in base al tipo selezionato
        if type == "input":
            title = "Correlazione tra Input"
            corr_matrix = input_rows_df_atot_max.drop(columns=['ID']).corr()  # Calcola la correlazione solo sulle colonne di input ripetute senza l'ID
            figsize=(10, 8)
        elif type == "input-output_03":
            title = "Correlazione tra Output a t=0.3 secondi"
            corr_matrix = result_df[[col for col in self.input_columns if col != "ID"] + self.output_03].corr().loc[[col for col in self.input_columns if col != "ID"], self.output_03]
            plt.xlabel("Output_t=0.3sec")
            plt.ylabel("Input")
            figsize=(10, 8)          
        elif type == "output_03":
             title = "Correlazione tra Output a t=0.3 secondi"
             # Filtrare self.output_03 escludendo 'U1_CL' e 'U3_CL'
             filtered_outputs = [col for col in self.output_03 if col not in ['U1_CL', 'U3_CL']]
             # Calcolare la matrice di correlazione con le colonne filtrate
             corr_matrix = result_df[['ID'] + filtered_outputs].corr().loc[filtered_outputs, filtered_outputs]
             plt.xlabel("Output_t=0.3sec")
             plt.ylabel("Output_t=0.3sec")
             figsize = (10, 8)
        elif type == "input-output_atot":
            title = "Correlazione tra Input e Output all'impatto"
            corr_matrix = merged_df_atot_max[[col for col in self.input_columns if col != "ID"] + self.output_atot_max].corr().loc[[col for col in self.input_columns if col != "ID"], self.output_atot_max]
            plt.xlabel("Output all'impatto")
            plt.ylabel("Input")
            figsize=(10, 8)
        elif type == "out_a_tot_max":
            title = "Correlazione tra Output all'impatto"
            corr_matrix = merged_df_atot_max[['ID'] + self.output_atot_max].corr().loc[self.output_atot_max, self.output_atot_max]
            plt.xlabel("Output all'impatto")
            plt.ylabel("Output all'impatto")
            figsize=(12, 8)
        else:
            print(f"Tipo di correlazione non riconosciuto: {type}")
            return  # Interrompe l'esecuzione della funzione
        '''
        elif type == "output":
          title = "Correlazione tra Output"
          corr_matrix = merged_df[self.output_columns].corr().loc[self.output_columns, self.output_columns]
          #xticklabels = self.output_columns
          #yticklabels = self.output_columns
          plt.xlabel("Output")
          plt.ylabel("Output")
          figsize=(12, 8)
       elif type == "in-out":
          title = "Correlazione tra Input e Output"
          corr_matrix = merged_df[[col for col in self.input_columns if col != "ID"] + self.output_columns].corr().loc[[col for col in self.input_columns if col != "ID"], self.output_columns]
          #xticklabels = self.output_columns
          #yticklabels = self.input_columns
          plt.xlabel("Output")
          plt.ylabel("Input")
          figsize=(12,8)
        '''
        
     # Configura la figura e salva l'immagine
        plt.figure(figsize=figsize)  # Imposta la dimensione predefinita per il grafico
        sns.heatmap(corr_matrix, annot=True, cmap="jet", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
        plt.title(title)

        # Salva il grafico
        file_name = f'correlazione_{type}.png'
        filepath = os.path.join('OUTPUT', file_name)
        plt.savefig(filepath, dpi=300)  # Imposta una risoluzione alta per l'immagine
        plt.close()  # Chiude la figura per evitare sovrapposizioni

        print(f"Grafico {title} salvato in {filepath}")
        
#def main():
limit_row_in_secondi = 0.3

# unione file e generazione calcoli
parametri = get_parametri()

diz = merged_file(parametri,limit_row_in_secondi )

create_xls_merge(copy.deepcopy(diz))

# Machine Learning
#TRAINING
corr = CorrelationAndPlotter(diz)  # Oggetto della classe

#CORRELAZIONI
corr.calculate_and_plot_correlations(type="input", show=True)
corr.calculate_and_plot_correlations(type="input-output_03", show=True)
corr.calculate_and_plot_correlations(type="output_03", show=True)
corr.calculate_and_plot_correlations(type="input-output_atot", show=True)
corr.calculate_and_plot_correlations(type="out_a_tot_max", show=True)
