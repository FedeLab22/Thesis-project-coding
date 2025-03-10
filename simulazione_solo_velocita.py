# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:19:09 2025

@author: fefe2
"""
# Importazione delle librerie necessarie
import pandas as pd
import numpy as np
import torch, copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('TkAgg')
from merged import merged_file
from merged import get_parametri
#from merged import create_xls_merge
from merged import create_xls_output
#import math

c= 2759
limit_row_in_secondi = 0.3

class LSTMTrainer:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        # Definire le colonne di input e output esplicitamente per titolo
        self.input_columns =  [ 'tempo','Volume', 'Vel trasl blocco', 'Angolo impatto', 'Pos impatto Z', ]
        self.output_columns = [  'U1_T_Block', 'U2_T_Block',  'U3_T_Block', 'V1_T_Block', 'V2_T_Block', 'V3_T_Block', 'V_T_Block', 'V1_R_Block', 'V2_R_Block', 'V3_R_Block',  'V_R_Block']
        
        self.list_excludes = ['Status','tempo_units' , 'RM1_MSX_units', 'RF3_MDX_units', 'U3_CAVOSUP_units', 'U2_T_Block_units', 'KE_Block_units', 'A3_T_Block_units', 'A1_T_Block_units', 'V3_T_Block_units', 'RF1_MDX_units', 'U2_CAVOSUP_units', 'RM3_MDX_units', 'RF3_MSX_units', 'RF2_MSX_units', 'RF1_MSX_units', 'V1_R_Block_units', 'U3_T_Block_units', 'U1_CAVOSUP_units', 'RM2_MSX_units', 'RM3_MSX_units', 'V3_R_Block_units', 'A2_T_Block_units', 'V2_R_Block_units', 'V1_T_Block_units', 'RM1_MDX_units', 'RF2_MDX_units', 'V2_T_Block_units', 'U1_T_Block_units', 'RM2_MDX_units','' ]

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.data_filled = None
        
        self.max_length = 0
        self.prediction = []
        
        # Inizializziamo gli scaler come attributi della classe
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()


    def preprocess_data(self):
        """
        Preprocess the data from the provided dictionary.
        Handles lists of varying lengths by padding with NaN.
        """
        self.data_filled = pd.DataFrame()  # Inizializza DataFrame vuoto

        # Trova la lunghezza massima delle liste
        for key, value in self.data_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.max_length = max(self.max_length, len(sub_value))

        # Normalizza le lunghezze delle liste --> KEY = ID, value = valori in un ID univoco
        for key, value in self.data_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list) and len(sub_value) < self.max_length:
                        value[sub_key] = sub_value + [np.nan] * (self.max_length - len(sub_value))

                # Rimuoviamo le righe nella lista di esclusioni
                for sub_key in list(value.keys()):  # Evita modifiche mentre itera
                    if sub_key in self.list_excludes:
                        del value[sub_key]  # Elimina le chiavi escluse

            # Crea il DataFrame per questa parte dei dati
            df = pd.DataFrame(value)

            # Concatena direttamente in self.data_filled
            self.data_filled = pd.concat([self.data_filled, df], ignore_index=True)
            
        # Verifica la presenza delle colonne necessarie
        missing_inputs = [col for col in self.input_columns if col not in self.data_filled.columns]
        missing_outputs = [col for col in self.output_columns if col not in self.data_filled.columns]
        if missing_inputs or missing_outputs:
            raise ValueError(f"Colonne mancanti! Input: {missing_inputs}, Output: {missing_outputs}")

        # Sostituisci NaN con la media delle colonne
        self.data_filled = self.data_filled.fillna(self.data_filled.mean())
        #self.data_filled = self.data_filled.apply(lambda x: x.fillna(x.mean()) if x.dtype in ['float64', 'int64'] else x)

        # Rimuovi colonne con deviazione standard zero
        # Seleziona solo le colonne numeriche
        self.data_filled = self.data_filled.select_dtypes(include=['number'])
        # Mantieni solo le colonne con deviazione standard diversa da 0
        self.data_filled = self.data_filled.loc[:, self.data_filled.std() != 0]

        X = self.data_filled[self.input_columns].values
        y = self.data_filled[self.output_columns].values

        # Fittiamo gli scaler una volta sola e li salviamo nella classe
        self.scaler_X.fit(X)
        self.scaler_y.fit(y)

        # Trasformiamo i dati
        X_normalized = self.scaler_X.transform(X)
        y_normalized = self.scaler_y.transform(y)

        return X_normalized, y_normalized

    def prepare_data_loaders(self, X, y):
        X, y = self.preprocess_data()  # Ora chiamiamo direttamente la funzione che normalizza
        dataset = RockImpactDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    def define_model(self, model_type="LSTM"):
        input_size = len(self.input_columns)
        output_size = len(self.output_columns)
        hidden_size = 256
        num_layers = 4  # Aumentiamo gli strati
        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers)

    def train_model(self, epochs=25):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        train_losses, val_losses = [], []
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in self.train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(self.train_loader))

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    val_predictions = self.model(X_val)
                    val_loss += criterion(val_predictions, y_val).item()
            val_losses.append(val_loss / len(self.val_loader))

            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")

        return train_losses, val_losses

    def time_simulate_output(self, limit_row_in_secondi, Volume, Vel_trasl_blocco, Angolo_impatto, Pos_impatto_Z):
        """
        Simula l'output per una serie di input, normalizzandoli prima di passarli alla rete neurale,
        e denormalizzando il risultato per ottenere dati reali.
        """
        self.prediction = []  # Lista per salvare i risultati

        # Genera i valori di tempo per la simulazione
        for tempo in np.linspace(0, limit_row_in_secondi, self.max_length):
            custom_input = [[tempo, Volume, Vel_trasl_blocco, Angolo_impatto, Pos_impatto_Z]]

            # Normalizziamo l'input con lo stesso scaler usato per il training
            new_input_normalized = self.scaler_X.transform(custom_input)

            # Convertiamo in tensore per PyTorch
            input_tensor = torch.tensor(new_input_normalized, dtype=torch.float32).unsqueeze(0)

            # Effettuiamo la previsione con la rete neurale
            self.model.eval()
            with torch.no_grad():
                output_tensor = self.model(input_tensor).numpy().reshape(-1, len(self.output_columns))

            # Denormalizziamo l'output per ottenere valori reali
            output_real = self.scaler_y.inverse_transform(output_tensor)

            # Salviamo solo i dati reali finali
            self.prediction.append((output_real, custom_input))

        return self.prediction  # Ritorna i risultati reali
    
    def print_simulation(self):
        simulation_df=[]
        for prediction in self.prediction:
            output, custom_input = prediction
            
            # Crea un DataFrame per l'output
            sdf_output = pd.DataFrame(output, columns=self.output_columns)

            # Crea un DataFrame per l'input
            sdf_input = pd.DataFrame(custom_input, columns=self.input_columns)

            # Combina input e output
            sdf_combined = pd.concat([sdf_input,sdf_output, ], axis=1)
            simulation_df.append(sdf_combined)
        final_df = pd.concat(simulation_df, axis=0).reset_index(drop=True)
        final_dict =final_df.to_dict(orient='list')

        return final_dict

class RockImpactDataset(Dataset):
    def __init__(self, X, y, time_steps=62):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.time_steps = time_steps

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_seq = self.X[idx].unsqueeze(0).repeat(self.time_steps, 1)
        output_seq = self.y[idx]
        return input_seq, output_seq

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


limit_row_in_secondi = 0.3

print("\nInserisci i valori di input.\n")
volume = float(input("Volume: ").replace(',','.'))
vel_trasl_blocco = float(input("Velocità: ").replace(',','.'))
angolo_impatto = float(input("Angolo impatto: ").replace(',','.'))
pos_impatto_Z = float(input("Posizione impatto: ").replace(',','.'))


# unione file e generazione calcoli
parametri = get_parametri()

diz = merged_file(parametri, limit_row_in_secondi)

#create_xls_merge(copy.deepcopy(diz))

# Machine Learning
#TRAINING
trainer = LSTMTrainer(diz)
X, y = trainer.preprocess_data()
trainer.prepare_data_loaders(X, y)

trainer.define_model()

trainer.train_model(epochs=25)

#SIMULAZIONE
trainer.time_simulate_output(limit_row_in_secondi, volume, vel_trasl_blocco, angolo_impatto, pos_impatto_Z)
diz = trainer.print_simulation()
create_xls_output(copy.deepcopy(diz))
#trainer.plot_simulation()

for param in trainer.model.parameters():
    if torch.isnan(param).any():
        print("Attenzione: Il modello contiene pesi NaN!")
