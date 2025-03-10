# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:46:02 2024

@author: fefe2
"""

import pandas as pd
from openpyxl import load_workbook
import os

# Percorso dei file
input_file_txt = os.path.join("INPUT","parametri.txt")

def get_parametri():
    # Leggi i dati dal file TXT (parametri)
    parametri_df = pd.read_csv(input_file_txt, sep=";", header=None)
  
    # Per ogni ID univoco, assegna i valori dei parametri corrispondenti
    parametri_columns = ['Volume',"Vel trasl blocco","","Angolo impatto","","Pos impatto Z"] #[f"Input_{i+1}" for i in range(parametri_df.shape[1])]  # Nomi delle colonne dei parametri
    parametri_df.columns = parametri_columns

    # Crea un dizionario con gli ID univoci come chiavi e i parametri come valori
    #result_dict =  parametri_df.to_dict()
    result_dict = {
        int(idx+1): parametri_df.iloc[idx].to_dict()
        for idx, unique_id in enumerate( parametri_df.iloc[:, 0].tolist() )
    }

    #print(result_dict)
    #print("Dizionario parametri generato con successo.") 

    # Restituisci il dizionario
    return result_dict