# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 19:30:54 2025

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
import math

c= 2759
limit_row_in_secondi = 0.3


class PhysicsLoss(nn.Module):
    def __init__(self, lambda_energy=0.1, lambda_momentum=0.1):
        super(PhysicsLoss, self).__init__()
        self.lambda_energy = lambda_energy
        self.lambda_momentum = lambda_momentum
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)

        # Conservazione dell'energia: KE_BLOCK ≈ E_TRASL + E_ROT
        energy_constraint = torch.sum((y_pred[:, -1] - (y_pred[:, -3] + y_pred[:, -2]))**2)


        total_loss = mse_loss + self.lambda_energy * energy_constraint

        return total_loss

class LSTMTrainer:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        # Definire le colonne di input e output esplicitamente per titolo
        self.input_columns =  [ 'tempo','Volume', 'Vel trasl blocco', 'Angolo impatto', 'Pos impatto Z', ]
        self.output_columns = [ 'U1_T_Block', 'U2_T_Block',  'U3_T_Block', 'V1_T_Block', 'V2_T_Block', 'V3_T_Block', 'V_T_Block', 'V1_R_Block', 'V2_R_Block', 'V3_R_Block', 'V_R_Block', 'A1_T_Block','A2_T_Block', 'A3_T_Block', 'A_T_Block', 'A_TOT',  'KE_Block',  'E_TRASL', 'E_ROT', 'RF_MDX', 'RF_MSX', 'RM_MDX', 'RM_MSX', 'FRECCIA_MAX_CAVOSUP',   ]
        self.list_excludes = ['Status','tempo_units' , 'RM1_MSX_units', 'RF3_MDX_units', 'U3_CAVOSUP_units', 'U2_T_Block_units', 'KE_Block_units', 'A3_T_Block_units', 'A1_T_Block_units', 'V3_T_Block_units', 'RF1_MDX_units', 'U2_CAVOSUP_units', 'RM3_MDX_units', 'RF3_MSX_units', 'RF2_MSX_units', 'RF1_MSX_units', 'V1_R_Block_units', 'U3_T_Block_units', 'U1_CAVOSUP_units', 'RM2_MSX_units', 'RM3_MSX_units', 'V3_R_Block_units', 'A2_T_Block_units', 'V2_R_Block_units', 'V1_T_Block_units', 'RM1_MDX_units', 'RF2_MDX_units', 'V2_T_Block_units', 'U1_T_Block_units', 'RM2_MDX_units','' ]
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.data_filled = None
        
        self.max_length = 0
        self.prediction = []

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def preprocess_data(self):
        # Preprocess the data from the provided dictionary.
        # Handles lists of varying lengths by padding with NaN.
        
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
        
        # Calcolo del lato a e dell'inerzia I
        self.data_filled['a'] = (3 * self.data_filled['Volume'] / (12 + 10 * math.sqrt(2))) ** (1/3)  # Calcolo del lato a
        self.data_filled['I'] = (1/6) * self.data_filled['Volume'] * c * self.data_filled['a']**2  # Correzione con ** per elevamento a potenza
        
        # Aggiorniamo le colonne di input per includere 'I' e 'a'
        self.input_columns += ['I', 'a']
        
        #Aggiungo l'energia 
        self.data_filled['KE_Block_CALC'] = 0.5 * self.data_filled['Volume'] * c * (self.data_filled['V_T_Block']**2 + self.data_filled['V_R_Block']**2)
        self.data_filled['E_trasl_CALC'] = 0.5 * self.data_filled['Volume'] * c * (self.data_filled['V_T_Block']**2)
        self.data_filled['E_ROT_CALC'] = self.data_filled['KE_Block_CALC'] - self.data_filled['E_trasl_CALC']
        self.data_filled['E_ROT_Ia'] = (1/12) * self.data_filled['Volume'] * c * self.data_filled['a']**2 * self.data_filled['V_R_Block']**2
        #A_TOT_CALC = 9.81
        
        # Aggiorniamo le colonne di input per includere 'I' e 'a'
        self.output_columns += ['KE_Block_CALC', 'E_trasl_CALC', 'E_ROT_CALC', 'E_ROT_Ia' ]
        
        
        # Verifica la presenza delle colonne necessarie
        missing_inputs = [col for col in self.input_columns if col not in self.data_filled.columns]
        missing_outputs = [col for col in self.output_columns if col not in self.data_filled.columns]
        if missing_inputs or missing_outputs:
            raise ValueError(f"Colonne mancanti! Input: {missing_inputs}, Output: {missing_outputs}")

        # Sostituisci NaN con la media delle colonne
        self.data_filled = self.data_filled.apply(lambda x: x.fillna(x.mean()) if x.dtype in ['float64', 'int64'] else x)
        self.data_filled = self.data_filled.fillna(0)  # Per le colonne con solo NaN

        # Aggiungi eventuali colonne mancanti per evitare errori
        for col in self.input_columns + self.output_columns:
            if col not in self.data_filled.columns:
                self.data_filled[col] = np.nan  # Inizializza con NaN per evitare errori

        # Riordina le colonne per evitare errori
        self.data_filled = self.data_filled[self.input_columns + self.output_columns]

        # Separa input e output
        X = self.data_filled[self.input_columns].values
        y = self.data_filled[self.output_columns].values

        # Normalizzazione
        self.scaler_X.fit(X)  # Calcola min/max degli input
        self.scaler_y.fit(y)  # Calcola min/max degli output

        X = self.scaler_X.transform(X)  # Trasforma gli input
        y = self.scaler_y.transform(y)  # Trasforma gli output

        # Salva i dati normalizzati
        self.X = X  
        self.y = y

        return X, y

    def prepare_data_loaders(self, X, y):
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
        
    def train_model(self, epochs=10):
        criterion =  PhysicsLoss(lambda_energy=0.1, lambda_momentum=0.1)
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

    def time_simulate_output(self, limit_row_in_secondi, volume, vel_trasl_blocco, angolo_impatto, pos_impatto_Z):
        # Calcolo del lato del rombicubottaedro (a) e dell'inerzia (I) prima della normalizzazione
        a = (3 * volume / (12 + 10 * math.sqrt(2))) ** (1/3)  # Calcolo del lato a
        I = (1/6) * c * volume * a**2  # Calcolo dell'inerzia (I), ora con c definito

        # Generazione degli input normalizzati
        new_input_data = np.array([[0, volume, vel_trasl_blocco, angolo_impatto, pos_impatto_Z, I, a]])
        new_input_normalized = self.scaler_X.transform(new_input_data)

        self.prediction = []
        for i in range(self.max_length):
            # Usa dati pre-elaborati se disponibili, altrimenti usa il nuovo input normalizzato
            input_data_normalized = self.X[i].reshape(1, -1) if i < len(self.X) else new_input_normalized
            
            input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32).unsqueeze(0)

            # Modello in modalità valutazione
            self.model.eval()
            with torch.no_grad():
                output_tensor = self.model(input_tensor).numpy().reshape(-1, len(self.output_columns))

            # Salvataggio dei risultati (valori normalizzati)
            self.prediction.append((output_tensor, input_data_normalized))

    def print_simulation(self):
        """
        Stampa i risultati della simulazione, unendo gli input e gli output
        """
        scaler_X = self.scaler_X  # Scaler per denormalizzare gli input
        scaler_y = self.scaler_y  # Scaler per denormalizzare gli output

        simulation_df = []
        denormalized_predictions = []

        # Denormalizzazione dei risultati
        for output_tensor, input_data_normalized in self.prediction:
            output_tensor_real = scaler_y.inverse_transform(output_tensor)
            input_data_real = scaler_X.inverse_transform(input_data_normalized) if input_data_normalized is not None else None
            denormalized_predictions.append((output_tensor_real, input_data_real))

        # Creazione del DataFrame finale
        for output, input_real in denormalized_predictions:
            sdf_output = pd.DataFrame(output, columns=self.output_columns)
            sdf_input = pd.DataFrame(input_real, columns=self.input_columns)

            # Sostituisci NaN e riempi Volume
            sdf_input.iloc[1:, :] = np.nan  # Mette NaN su tutte le righe successive
            sdf_input["Volume"] = sdf_input.iloc[0]["Volume"]  # Riassegna il volume

            # Concatenazione degli input e output
            sdf_combined = pd.concat([sdf_input, sdf_output], axis=1)
            simulation_df.append(sdf_combined)

        final_df = pd.concat(simulation_df, axis=0).reset_index(drop=True)
        return final_df.to_dict(orient='list')


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

trainer.train_model(epochs=1)

#SIMULAZIONE
trainer.time_simulate_output(limit_row_in_secondi, volume, vel_trasl_blocco, angolo_impatto, pos_impatto_Z)
diz = trainer.print_simulation()
create_xls_output(copy.deepcopy(diz))
#trainer.plot_simulation()

for param in trainer.model.parameters():
    if torch.isnan(param).any():
        print("Attenzione: Il modello contiene pesi NaN!")
