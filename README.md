# seq2seq_electric
This provides the code for deep Recurrent Neural network for electric load forecasting

Instructions: The repo has 2 man files: lstm_106.py and lstm_106b.py. The first is for continous-type load profiles and the second for discrete-type.

Instructions for lstm_106.py:

(1) Save the folders 'SLC PSB' and 'WBB Weather Data' (provided in the repo).

(2) Go to file DataFunctions.py. In the file, Go the read_multiple_csv() function (line 290) and change the folder path to the location where the folder 'SLC PSB' was saved.

(3) Stay in file DataFunctions.py. In the file, Go the read_weather_files() function (line 308) and change the folder path to the location where the folder 'WBB Weather Data' was saved.

(4) Go to the main file 'lstm_106.py'. Pick one of the two load profiles: HVAC_critical or HVAC_normal (line 59).

(5) The variable 'train_data' is the 2-D array of feature data used for training. Rows represent data points and columns represent featurs. If you are doing a feature sensitivity analysis, pick a subset of these features.

(6) In line 93, keep 'choice==1'. This will load the energy consumption data after missing values have been interpolated for. 

(7) Pick a load pattern to load, depending on the selection of load pattern in (4). If in (4) you selected HVAC_critical, put in "H_t = numpy.load('H1_file_HVAC1.npy')". If in (4) you selected HVAC_normal, put in "H_t = numpy.load('H1_file_HVAC1.npy')"

(8) In line 150, pick 'H_val' as one of 'HVAC_critical' or 'HVAC_normal'. This has to be identical to the selection in (4).

(9) The variable 'val_data' is the 2-D array of feature data used for validation. Rows represent data points and columns represent featurs. If you are doing a feature sensitivity analysis, pick a subset of these features. Note: these features have to correspond to those chosen for 'train_data' in (5).

(10) In line 150, pick 'H_e' one of 'HVAC_critical' or 'HVAC_normal'. This has to be identical to the selection in (4) and (8).

(11) The variable 'test_data' is the 2-D array of feature data used for test. Rows represent data points and columns represent featurs. If you are doing a feature sensitivity analysis, pick a subset of these features. Note: these features have to correspond to those chosen for 'train_data' in (5).

(12) Run 'lstm_106.py'.
