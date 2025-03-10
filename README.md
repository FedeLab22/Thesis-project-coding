"parametri.txt": contain all the combination of input parameters.

"L3_input_espansi.py": allow to import inputa data ("parametri.txt") in a unic file .xlsx as input for ML.
"output_to_merged.py": this code create the final file.xlsx which contains all the abaqus results.
"export_xlsx.py": it is useful to print the file "output_merged.xlsx" created merging the previous one and then the "simulazioni.xlsx" files.

"main.py" is useful but not essential for calling up and launching the other codes with a single script. Currently each script works individually;

CORRELATION MATRIX
"Correlazioni.py": here the implementation to create correlation matrix.
All the "files.png" represent the results of global sensitivity analysis

PREDICTIVE MODEL
"N_ML_neuralnetwork.py": the code shows the implementation of the winning strategy in which the physical relationships are worked out following the simulation;
"train_model.py": here is implemented the version where regression is set in training phase;
"simulazione_solo_velocita.py": here there are less values in pre-processing phase.

all the excel files represent the results of these codes part of "Predictive model".
