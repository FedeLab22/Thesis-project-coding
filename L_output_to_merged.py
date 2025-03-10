import os
import pandas as pd
import re  # Importa il modulo per le espressioni regolari
from json.decoder import NaN
import numpy as np

# Percorso principale della cartella di input
input_folder =  os.path.join("INPUT","Output")

# Lista dei pattern da escludere
excluded_patterns = ["A1_R_Block", "A2_R_Block", "A3_R_Block", "U1_R_Block", "U2_R_Block", "U3_R_Block", "Area_Contact_Block"]


# costante
c=2759


# Mappatura esplicita dei gruppi per ogni tipo
expected_groups = {
    "FRECCIA_MAX_CAVOSUP": ["U1_CAVOSUP", "U2_CAVOSUP", "U3_CAVOSUP"],
    "A_T_Block": ["A1_T_Block", "A2_T_Block", "A3_T_Block"],
    "V_T_Block": ["V1_T_Block", "V2_T_Block", "V3_T_Block"],
    "V_R_Block": ["V1_R_Block", "V2_R_Block", "V3_R_Block"],
    "RF_MDX": ["RF1_MDX", "RF2_MDX", "RF3_MDX"],
    "RF_MSX": ["RF1_MSX", "RF2_MSX", "RF3_MSX"],
    "RM_MDX": ["RM1_MDX", "RM2_MDX", "RM3_MDX"],
    "RM_MSX": ["RM1_MSX", "RM2_MSX", "RM3_MSX"],
}

# Unità di misura per ciascun gruppo
units_mapping = {
    "FRECCIA_MAX_CAVOSUP": "[m]",
    "A_T_Block": "[m/s^2]",
    "V_T_Block": "[m/s]",
    "V_R_Block": "[m/s]",
    "RF_MDX": "[N]",
    "RF_MSX": "[N]",
    "RM_MDX": "[N·m]",
    "RM_MSX": "[N·m]",
}

# Controllo che la cartella di output esista, altrimenti la creo
#os.makedirs(output_folder, exist_ok=True)

# Funzione per determinare l'unità di misura in base al nome
def get_unit_from_filename(filename):
    if "A" in filename:
        return "m/s^2"
    elif "U" in filename:
        return "m"
    elif "V" in filename:
        return "m/s"
    elif "R" in filename:
        return "N"
    elif "KE" in filename:
        return "J"
    else:
        return ""

def merged_file(parametri, limit_row_in_secondi):
    diz = {}

    # Iterazione delle cartelle
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Controlla se il file "elements_fail" è presente
        fail_file = os.path.join(folder_path, f"{folder_name}_elements_fail.txt")
        idx = int(folder_name.split("_", 3)[-1].replace('ID', ''))  # Ottiene la parte ID

        if idx not in diz.keys():
            diz[idx] = {}

        if os.path.exists(fail_file):
            # Caso FAILURE: assegna Status = 0 e limit_row = 1
            diz[idx]['Status'] = [0]
            limit_row = 1

            # Salva comunque i parametri di input
            for k, v in parametri[idx].items():
                diz[idx][k] = [v]  # Salviamo una sola riga

            diz[idx]['ID'] = [idx]  # Assegniamo l'ID
            continue  # SALTA IL RESTO DEL CODICE

        # Inizializzazione della tabella per il caso di SUCCESSO
        merged_data = {"tempo": []}
        units = {"tempo": "s"}

        # Iterazione dei file nella cartella
        for file_name in os.listdir(folder_path):
            if any(pattern in file_name for pattern in excluded_patterns):
                continue

            file_path = os.path.join(folder_path, file_name)

            try:
                df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python')

                param_name = re.sub(r"^Job_60_13m_ID\d+_", "", file_name).replace(".txt", "")

                if df.shape[1] < 2:
                    continue

                units[param_name] = get_unit_from_filename(param_name)

                if merged_data["tempo"] == []:
                    limit_row = 0
                    for val in df.iloc[:, 0]:
                        if float(val) > limit_row_in_secondi:
                            break
                        limit_row += 1

                    merged_data["tempo"] = df.iloc[:limit_row, 0].tolist()

                merged_data[param_name] = df.iloc[:limit_row, 1].tolist()

            except Exception as e:
                print(f" *** Errore durante la lettura del file {file_path}: {e}")

        # Creazione del DataFrame finale per SUCCESSO
        try:
            final_df = pd.DataFrame(merged_data)
            diz[idx] = final_df.to_dict(orient='list')

            # Imposta Status = 1 per i successi
            diz[idx]['Status'] = [1 for _ in range(limit_row)]
            diz[idx]['ID'] = [idx for _ in range(limit_row)]

        except ValueError as e:
            print(f" *** Errore durante la creazione del DataFrame per la cartella {folder_name}: {e}")

        except ValueError as e:
            print(f" *** Errore durante la creazione del DataFrame per la cartella {folder_name}: {e}")
    first_idx = None
    for idx in diz.keys():
        # aggiungiamo le colonne indicate negli indici della variabile "expected_groups"
        if not first_idx:
            first_idx = idx
        for group, keys in expected_groups.items():
            diz[idx][group] = [
                (sum(val ** 2 for val in vals) ** 0.5)
                for vals in zip(*(diz[idx][k] for k in keys if k in diz[idx]))
            ]
            
        #AGGIUNGERE TRAIETTORIA PER OTTENERE LA DIFFERENZA TRA CADUTA LIBERA E CON RETE  
        
        # aggiungiamo E_TRASL
        diz[idx]['E_TRASL'] = []
        for i in range(0, limit_row):
            if diz[idx].get('V_T_Block'):
                e_trasl = 0.5 * (diz[idx]['V_T_Block'][i] ** 2) * c * parametri[idx]['Volume']
                diz[idx]['E_TRASL'].append(e_trasl)
            else:
                diz[idx]['E_TRASL'].append(np.nan)

        # aggiungiamo E_ROT
        try:
            diz[idx]['E_ROT'] = [diz[idx]['KE_Block'][i] - diz[idx]['E_TRASL'][i] for i in range(0, limit_row)]
        except:
            diz[idx]['E_ROT'] = [np.nan for i in range(0, limit_row)]

        # aggiungiamo A_TOT
        diz[idx]['A_TOT'] = [0, ]
        for i in range(0, limit_row - 1):
            try:
                V3 = ((diz[idx]['V3_T_Block'][i + 1] - diz[idx]['V3_T_Block'][i]) / (diz[idx]['tempo'][i + 1] - diz[idx]['tempo'][i])) ** 2
                V2 = ((diz[idx]['V2_T_Block'][i + 1] - diz[idx]['V2_T_Block'][i]) / (diz[idx]['tempo'][i + 1] - diz[idx]['tempo'][i])) ** 2
                V1 = ((diz[idx]['V1_T_Block'][i + 1] - diz[idx]['V1_T_Block'][i]) / (diz[idx]['tempo'][i + 1] - diz[idx]['tempo'][i])) ** 2
                diz[idx]['A_TOT'].append((V3 + V2 + V1) ** 0.5)
            except:
                diz[idx]['A_TOT'].append(np.nan)
                continue

        # Prima creiamo le colonne con NaN
        for param in ['Vel trasl blocco', 'Angolo impatto', 'Pos impatto Z']:
            diz[idx][param] = [float('nan')] * limit_row

        for k, v in parametri[idx].items():
            if k == 'Volume':
                diz[idx][k] = [v] if os.path.exists(fail_file) else [v for _ in range(limit_row)]
            elif k in ['Vel trasl blocco', 'Angolo impatto', 'Pos impatto Z']:
                if k in diz[idx]:  
                    diz[idx][k][0] = v  
            else:
                diz[idx][k] = [v] if os.path.exists(fail_file) else [v] * limit_row

    '''
    # Assicuriamoci che entrambe le chiavi esistano e contengano liste non vuote
    if ("V_T_Block" in diz[first_idx] and isinstance(diz[first_idx]["V_T_Block"], list) and len(diz[first_idx]["V_T_Block"]) > 0 and 
          "Vel trasl blocco" in diz[first_idx] and isinstance(diz[first_idx]["Vel trasl blocco"], list) and len(diz[first_idx]["Vel trasl blocco"]) > 0):
          
          # Assegniamo il valore se tutto è corretto
          diz[first_idx]["V_T_Block"][0] = diz[first_idx]["Vel trasl blocco"][0]
    else:
          # Debugging dettagliato
          print(f"'V_T_Block' esiste in diz[first_idx]? {'V_T_Block' in diz[first_idx]}" )
          print(f"'Vel trasl blocco' esiste in diz[first_idx]? {'Vel trasl blocco' in diz[first_idx]}" )
          print(f"diz[first_idx][\"V_T_Block\"] = {diz[first_idx].get('V_T_Block')}")
          print(f"diz[first_idx][\"Vel trasl blocco\"] = {diz[first_idx].get('Vel trasl blocco')}")
          print("Errore: 'V_T_Block' o 'Vel trasl blocco' non esiste o non contiene una lista valida.")
    '''
    return diz
