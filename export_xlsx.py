import pandas as pd
import os
from collections import OrderedDict
import numpy as np

def create_xls_output(diz, filename="simulazione.xlsx"):
    """
    Crea un file Excel dalla simulazione e lo formatta correttamente.
    Aggiunge le nuove colonne  V_T_Block_diretto.
    """
    # Creazione del DataFrame dalla simulazione
    simulation_df = pd.DataFrame.from_dict(diz)

    # Aggiungi indici temporali
    simulation_df.index = range(1, simulation_df.shape[0] + 1)

    # Gestisci i NaN dopo il primo istante per alcune colonne (se necessario)
    for col in ["Vel trasl blocco", "Angolo impatto", "Pos impatto Z", "I", "a"]:
        if col in simulation_df.columns:
            simulation_df.iloc[1:, simulation_df.columns.get_loc(col)] = np.nan

    # Creazione della cartella OUTPUT se non esiste
    output_folder = "OUTPUT"
    os.makedirs(output_folder, exist_ok=True)

    # Salvataggio in Excel
    output_file = os.path.join(output_folder, filename)
    simulation_df.to_excel(output_file, index=False)

    print(f"File Excel salvato correttamente in: {output_file}")

def create_xls_merge(diz,filename="output_merged.xlsx"):

    columns = [ 'ID', 'Status','Volume', 'Vel trasl blocco', 'Angolo impatto','Pos impatto Z', 'tempo', 'RF2_MSX', 'A1_T_Block', 'U2_CAVOSUP', 'U1_T_Block', 'RF1_MSX', 'A2_T_Block', 'U1_CAVOSUP', 'RM1_MDX', 'RM2_MDX', 'RM1_MSX', 'RF3_MSX', 'V1_T_Block', 'U3_T_Block', 'V2_R_Block', 'V3_R_Block', 'RF3_MDX', 'RM3_MDX', 'U2_T_Block', 'KE_Block', 'RM3_MSX', 'RF2_MDX', 'V3_T_Block', 'RF1_MDX', 'V2_T_Block', 'U3_CAVOSUP', 'V1_R_Block', 'RM2_MSX', 'A3_T_Block',   'FRECCIA_MAX_CAVOSUP', 'A_T_Block', 'V_T_Block', 'V_R_Block', 'RF_MDX', 'RF_MSX', 'RM_MDX', 'RM_MSX', 'E_TRASL', 'E_ROT', 'A_TOT',]

    key_units = ['tempo_units', 'RF2_MSX_units', 'A1_T_Block_units', 'U2_CAVOSUP_units', 'U1_T_Block_units', 'RF1_MSX_units', 'A2_T_Block_units', 'U1_CAVOSUP_units', 'RM1_MDX_units', 'RM2_MDX_units', 'RM1_MSX_units', 'RF3_MSX_units', 'V1_T_Block_units', 'U3_T_Block_units', 'V2_R_Block_units', 'V3_R_Block_units', 'RF3_MDX_units', 'RM3_MDX_units', 'U2_T_Block_units', 'KE_Block_units', 'RM3_MSX_units', 'RF2_MDX_units', 'V3_T_Block_units', 'RF1_MDX_units', 'V2_T_Block_units', 'U3_CAVOSUP_units', 'V1_R_Block_units', 'RM2_MSX_units', 'A3_T_Block_units']


    dati = [columns]
    try:
        units = ["","","m3","m/s","°","m",]
        for k,v in diz.items():
            for i in key_units :
                units.append(v[i])
            break

        dati.append(units)
    except:
        pass

    ordered_diz = OrderedDict(sorted(diz.items()))
    for k,v in ordered_diz.items():
        try:
            for i in range(0,len(v['tempo'])):
                values = []
                for c in columns:
                    try:
                        values.append(v[c][i])
                    except:
                        values.append("")
                dati.append(values)
        except:
             failure_values = [v.get(c, [""])[0] for c in ['ID', 'Status', 'Volume', 'Vel trasl blocco', 'Angolo impatto', 'Pos impatto Z']]
             dati.append(failure_values)

    # Crea un DataFrame
    df = pd.DataFrame(dati)
    percorso = os.path.join("OUTPUT",)
    if not os.path.exists(percorso):
        # Crea la cartella
        os.makedirs(percorso)

    # Scrivi il DataFrame in un file Excel
    output_file = os.path.join(percorso,filename)


    df.to_excel(output_file, index=False)

    print(f"File Excel creato: {output_file}")
