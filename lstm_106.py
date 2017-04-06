#importing keras modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input, Merge
from keras.layers import merge
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import fmin, tpe, hp
from tempfile import TemporaryFile

#importing graphics and numpy
import numpy
import pydot
print pydot.find_graphviz()

#importing Custom Libraries
#manual functions
import ArrayFunctions
import MathFunctions

import DataFunctions
import NNFunctions
import PlotFunctions
import InterpolateFunctions
import NNFun_PSB
import build_lstm_v1
#define hyperopt search space
import hyperopt.pyll.stochastic

from sklearn.metrics import r2_score
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_selection import chi2

#scipy library
from scipy.stats import pearsonr

seed = 7
numpy.random.seed(seed)

from datetime import  datetime

######## The actual code Starts here
#######Training data: 2015-16

#EnergyData
date_start = '5/19/15 12:00 AM' #Start Date for training
date_end = '5/13/16 11:59 PM' # End date for training ~roughly 1 year of training data
std_inv = 60 #in minutes

#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_t = DataFunctions.fix_data(HVAC_critical) #substract to find differences
H_t = DataFunctions.fix_energy_intervals(H_t, 5, std_inv) #convert to std_div time intervals
H_t = DataFunctions.fix_high_points(H_t)

#Weather Data for training
weather_file = DataFunctions.read_weather_files()
weather_train = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_train = DataFunctions.fix_weather_intervals(weather_train, 5, std_inv)
weather_train = DataFunctions.interpolate_nans(weather_train)

#feature vectors
X_sch_t = DataFunctions.compile_features(H_t, date_start, std_inv)

#
train_data = numpy.concatenate((weather_train[:, 0:2], X_sch_t), axis=1)
#train_data = X_sch_t
#train_data = DataFunctions.normalize_2D(train_data)

H_rms = MathFunctions.rms_flat(H_t)
#normalizing data
H_min, H_max = DataFunctions.get_normalize_params(H_t)
H_t = H_t/H_max

#Computing Entropy
S, unique, pk = DataFunctions.calculate_entropy(H_t)
print "The entropy value is: ", S

# Block to interpolate
cons_points = 5
s0 = 10
start_day = 15
end_day  = 16
s1 = 20

choice = 1

if choice == 1:
    H_t = numpy.load('H1_file_HVAC1.npy')
elif choice == 2:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(train_data, H_t)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_t = InterpolateFunctions.organize_pred(H_t, Y_p, test_list)
    #numpy.save('Ht_file_HVAC1.npy', H_t)
else:
    H1 = H_t.copy()
    small_list, large_list = InterpolateFunctions.interpolate_main(train_data, H_t, start_day, end_day, cons_points)
    H_t = InterpolateFunctions.interp_linear(H_t, small_list) #H_t is beign changed as numpy arrays are mutable
    H1 = InterpolateFunctions.interp_linear(H1, small_list)

    PlotFunctions.Plot_interp_params()
    H_t = H_t[:, None] #changing numpy array shape to fit the function
    Y_t, Y_NN = InterpolateFunctions.interp_LSTM(train_data, H_t, large_list)

    PlotFunctions.Plot_interpolate(H1[s0*24:start_day*24], Y_t[start_day*24:end_day*24], Y_NN[start_day*24:end_day*24], H1[start_day*24:end_day*24], H1[end_day*24:s1*24])
    e_interp = InterpolateFunctions.interpolate_calculate_rms(H1[start_day*24:end_day*24], Y_t[start_day*24:end_day*24])
    e_NN = InterpolateFunctions.interpolate_calculate_rms(H1[start_day*24:end_day*24], Y_NN[start_day*24:end_day*24])

    print e_interp
    print e_NN
    #H_t = Y_t.copy()
    H_t = Y_NN.copy()
    numpy.save('H1_file_HVAC1.npy', H_t)



#Aggregating data on a daily basis
print H_t.shape
H_t = H_t[:, None]
conv_hour_to_day = 24
H_mean_t, H_sum_t, H_min_t, H_max_t = DataFunctions.aggregate_data(H_t, conv_hour_to_day)
w_mean_t, w_sum_t, w_min_t, w_max_t = DataFunctions.aggregate_data(weather_train, conv_hour_to_day)

#gettomg features for a single day
#PlotFunctions.Plot_single(H_mean_t)
X_day_t = DataFunctions.compile_features(H_sum_t, date_start, 24*60)


######################################
########VALIDATION DATA
#################

#######
#Getting validation data
date_start = '5/12/16 12:00 AM'
date_end = '5/18/16 11:59 PM'

#Read data
#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_val = DataFunctions.fix_data(HVAC_critical) #substract to find differences
H_val = DataFunctions.fix_energy_intervals(H_val, 5, std_inv) #convert to std_div time intervals
H_val = DataFunctions.fix_high_points(H_val)


#Weather Data for training
weather_file = DataFunctions.read_weather_files()
weather_val = DataFunctions.read_weather_csv(weather_file, date_start, date_end)
weather_val = DataFunctions.fix_weather_intervals(weather_val, 5, std_inv)
weather_val = DataFunctions.interpolate_nans(weather_val)


#feature vectors
X_sch_val = DataFunctions.compile_features(H_val, date_start, std_inv)

val_data = numpy.concatenate((weather_val[:, 0:2], X_sch_val), axis=1)
#val_data = X_sch_val
#val_data = DataFunctions.normalize_2D(val_data)

#normalizing data
H_val = H_val/H_max
#X_sch_t = DataFunctions.normalize_vector(X_sch_t, X_min, X_max)

choice = 0
if choice == 1:
    H_val = numpy.load('Hv_file_total.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(val_data, H_val)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_val = InterpolateFunctions.organize_pred(H_val, Y_p, test_list)
    numpy.save('Hv_file_total.npy', H_val)


#Aggregating data on a daily basis
H_mean_v, H_sum_v, H_min_v, H_max_v = DataFunctions.aggregate_data(H_val, conv_hour_to_day)
w_mean_v, w_sum_v, w_min_v, w_max_v = DataFunctions.aggregate_data(weather_val, conv_hour_to_day)

#gettomg features for a single day
X_day_val = DataFunctions.compile_features(H_sum_v, date_start, 24*60)


##########################################################
###Test data: 2016
date_start = '5/19/16 12:00 AM'
date_end = '8/7/16 11:59 PM'


#Read data
#Read data
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_e = DataFunctions.fix_data(HVAC_critical) #substract to find differences
H_e = DataFunctions.fix_energy_intervals(H_e, 5, std_inv) #convert to std_div time intervals
H_e = DataFunctions.fix_high_points(H_e)

#Weather Data for training
weather_file = DataFunctions.read_weather_files()
weather_test = DataFunctions.read_weather_csv(weather_file, date_start, date_end) #sort by date and converts to a matrix format
weather_test = DataFunctions.fix_weather_intervals(weather_test, 5, std_inv)
weather_test = DataFunctions.interpolate_nans(weather_test)

#feature vectors
X_sch_e = DataFunctions.compile_features(H_e, date_start, std_inv)

#combining features
test_data = numpy.concatenate((weather_test[:, 0:2], X_sch_e), axis=1)
#test_data = X_sch_e
#test_data = DataFunctions.normalize_2D(test_data) #CUT THIS OFF FOR REGRESSION
#test_data = DataFunctions.normalize_2D(test_data)

#Normalize data
H_e = H_e/H_max
#X_sch_e = DataFunctions.normalize_vector(X_sch_e, X_min, X_max)

choice = 0

if choice == 1:
    H_e = numpy.load('He_file_total.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(test_data, H_e)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_e = InterpolateFunctions.organize_pred(H_e, Y_p, test_list)
    numpy.save('He_file_total.npy', H_e)



#Plotting imputated values
#PlotFunctions.Plot_double(train_list, Y_t, test_list, Y_p, 'Actual Value', 'Interpolated value', 'ro', 'bo')

#Aggregating data on a daily basis
H_mean_e, H_sum_e, H_min_e, H_max_e = DataFunctions.aggregate_data(H_e, conv_hour_to_day)
w_mean_e, w_sum_e, w_min_e, w_max_e = DataFunctions.aggregate_data(weather_test, conv_hour_to_day)

#gettomg features for a single day
X_day_e = DataFunctions.compile_features(H_sum_e, date_start, 24*60)

####
#Saving variables for MLP neural network
X1 = train_data
X2 = val_data
X3 = test_data

H1 = H_t
H2 = H_val
H3 = H_e
#  `
#print H_mean_t.shape

# Reshaping array into (#of days, 24-hour timesteps, #features)
train_data = numpy.reshape(train_data, (X_day_t.shape[0], 24, train_data.shape[1]))
val_data = numpy.reshape(val_data, (X_day_val.shape[0], 24, val_data.shape[1]))
test_data = numpy.reshape(test_data, (X_day_e.shape[0], 24, test_data.shape[1]))

X_sch_t = numpy.reshape(X_sch_t, (X_day_t.shape[0], 24, X_sch_t.shape[1]))
X_sch_val = numpy.reshape(X_sch_val, (X_day_val.shape[0], 24, X_sch_val.shape[1]))
X_sch_e = numpy.reshape(X_sch_e, (X_day_e.shape[0], 24, X_sch_e.shape[1]))
#H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24, 1))
#H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24, 1))

H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24))
H_val = numpy.reshape(H_val, (H_mean_v.shape[0], 24))
H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24))

#This block is for optimizing LSTM layers
space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 20, 1),
        'activ_l3': hp.choice('activ_l3', ['relu', 'sigmoid'])
         #'D1': hp.uniform('D1', 0, 0.5),
         #'D2': hp.uniform('D2', 0, 0.5),
         #'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
         }


print H_t
print H_val
print H_e

def objective(params):
    #optimize_model = build_lstm_v1.lstm_model_102(params, train_data.shape[2], 24, 24)
    #optimize_model = build_lstm_v1.lstm_model_106(params, train_data.shape[2], 24)
    optimize_model = build_lstm_v1.lstm_model_109(params, train_data.shape[2], 24)

    #for epochs in range(5):
    for ep in range(20):
        #optimize_history = optimize_model.fit(X_seq, Y_seq, batch_size=1, nb_epoch=3, validation_split=(X_seq, Y_seq), shuffle=False)
        optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_data=(val_data, H_val), shuffle=False)
        #optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.3, shuffle=False)
        optimize_model.reset_states()

    loss_v = optimize_history.history['val_loss']
    print loss_v

    loss_out = loss_v[-1]

    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=20)

#Building Stateful Model
lstm_hidden = hyperopt.space_eval(space, best)
print lstm_hidden
tsteps = 24
out_dim = 24

#lstm_model = build_lstm_v1.lstm_model_102(lstm_hidden, train_data.shape[2], out_dim, tsteps)
lstm_model = build_lstm_v1.lstm_model_109(lstm_hidden, train_data.shape[2], tsteps)
save_model = lstm_model

##callbacks for Early Stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

#parameters for simulation
attempt_max = 5
epoch_max = 200
min_epoch = 20

#Criterion for early stopping
tau = 10
e_mat = numpy.zeros((epoch_max, attempt_max))
e_temp = numpy.zeros((tau, ))

tol = 0
count = 0
val_loss_v = []
epsilon = 1 #initialzing error
loss_old = 1
loss_val = 1

for attempts in range(attempt_max):
    lstm_model = build_lstm_v1.lstm_model_109(lstm_hidden, train_data.shape[2], tsteps)
    print "New model Initialized"

    for ep in range(epoch_max):
        lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0, shuffle=False)

        loss_old = loss_val
        loss_val = lstm_history.history['loss']

        # testing alternative block
        #lstm_model.reset_states()
        y_val = lstm_model.predict(val_data, batch_size=1, verbose=0)
        e1, e2 = DataFunctions.find_error(H_t, H_val, y_val)
        print e1, e2
        val_loss_check = e2
        val_loss_v.append(val_loss_check)
        e_mat[ep, attempts] = val_loss_check

        if val_loss_v[count] < epsilon and loss_val < loss_old:
            epsilon = val_loss_v[count]
            save_model = lstm_model
            test_model = lstm_model
            Y_lstm = test_model.predict(test_data, batch_size=1, verbose=0)
            e_1, e_2 = DataFunctions.find_error(H_t, H_e, Y_lstm)
            test_model.reset_states()
            print e_1
            print e_2

        count = count + 1
        lstm_model.reset_states()


        #This block is for early stopping
        if ep>=min_epoch:
            e_temp = e_mat[ep - tau + 1: ep + 1, attempts]
            e_local = e_mat[ep-tau, attempts]

            #print e_temp
            #print e_local

            if numpy.all(e_temp > e_local):
                break



        #if val_loss_check < tol:
            #break


print val_loss_v

#Y_lstm = lstm_model.predict(test_data, batch_size=1, verbose=0)
#Y1 = save_model.predict(train_data, batch_size=1, verbose=0) #get the states up to speed
#Y2 = save_model.predict(val_data, batch_size=1, verbose=0) #get the states up to speed
Y_lstm2 = save_model.predict(test_data, batch_size=1, verbose=0)
#numpy.save('Y_file_CONV1.npy', Y_lstm2)

#### Error analysis
H_t = numpy.reshape(H_t, (H_t.shape[0]*24, 1))
H_e = numpy.reshape(H_e, (H_e.shape[0]*24, 1))
Y_lstm = numpy.reshape(Y_lstm, (Y_lstm.shape[0]*24, 1))
Y_lstm2 = numpy.reshape(Y_lstm2, (Y_lstm2.shape[0]*24, 1))
t_train = numpy.arange(0, len(H_t))
t_test = numpy.arange(len(H_t), len(H_t)+len(Y_lstm2))
t_array = numpy.arange(0, len(Y_lstm2))

e_deep = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_e))
e_deep2 = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_t))
#e_deep3 = (MathFunctions.rms_flat(Y_lstm - H_e))/(MathFunctions.rms_flat(H_e))
#e_deep4 = (MathFunctions.rms_flat(Y_lstm - H_e))/(MathFunctions.rms_flat(H_t))



print e_deep
print e_deep2
#print e_deep3
#print e_deep4


### Reshape arrays for daily neural network
X_day_t = numpy.reshape(X_day_t, (X_day_t.shape[0], 1, X_day_t.shape[1]))
X_day_e = numpy.reshape(X_day_e, (X_day_e.shape[0], 1, X_day_e.shape[1]))
H_day_t = numpy.concatenate((H_mean_t, H_max_t, H_min_t), axis=1)
H_day_e = numpy.concatenate((H_mean_e, H_max_e, H_min_e), axis=1)



#### Implement MLP Neural Network
best_NN = NNFunctions.NN_optimizeNN_v21(X1, H1, X2, H2)
NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
NN_savemodel = NN_model

epsilon = 1
val_loss_v = []

for attempts in range(0, 5):
    NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
    NN_history = NN_model.fit(X1, H1, validation_data=(X1, H1), nb_epoch=50, batch_size=1, verbose=0, callbacks=callbacks)

    loss_v = NN_history.history['val_loss']
    val_loss_check = loss_v[-1]
    #print val_loss_check
    val_loss_v.append(val_loss_check)

    if val_loss_v[attempts] < epsilon:
        epsilon = val_loss_v[attempts]
        NN_savemodel = NN_model



Y_NN = NN_savemodel.predict(X3)
#Y_NN = numpy.reshape(Y_NN, (Y_NN.shape[0]*24, 1))
e_NN = (MathFunctions.rms_flat(Y_NN - H3))/(MathFunctions.rms_flat(H3))
e_NN2 = (MathFunctions.rms_flat(Y_NN - H3))/(MathFunctions.rms_flat(H1))


print e_NN
print e_NN2
print "R2: "
print r2_score(Y_lstm2, H_e)

#Calculate p-value
a1, a2 = pearsonr(Y_lstm2, H_e)
print "LSTM rho-value: "
print a1

b1, b2 = pearsonr(Y_NN, H_e)
print "NN rho-value"
print b1

#### Plotting
PlotFunctions.Plot_double(t_array, H_e, t_array, Y_lstm2, 'Actual conv power','LSTM conv power', 'k-', 'r-', "fig_HVAC1a.eps")
PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm2, t_test, H_e, 'Training Data', 'Model B predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_HVACA1b.eps")
PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm, t_test, H_e, 'Training Data', 'Model B predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_HVAC1c.eps")
PlotFunctions.Plot_quadruple(t_train, H_t, t_test, Y_lstm2, t_test, Y_NN, t_test, H_e, 'Training Data', 'Model B predictions', 'MLP Predictions', 'Test Data (actual)', 'k-', 'r-', 'y-', 'b-', "fig_HVACA1d.eps")

#Saving files
numpy.save('HVAC_critical_B', Y_lstm2)