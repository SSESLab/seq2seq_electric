from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Merge, Input
from keras.utils.visualize_util import plot
from keras.models import Model
from keras.layers import merge
from keras.layers import LSTM
import math

from keras.callbacks import EarlyStopping
import numpy
import pydot
import keras
print pydot.find_graphviz()

#Cross validation
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler
import sys

#hyperas modules
import pymongo
import hyperas
import hyperopt
from hyperopt import Trials, STATUS_OK, tpe
from hyperopt import fmin, tpe, hp

#custom libraries
import build_lstm_v1
import MathFunctions

#Function to optimize Real and Binary Schedules

def OptimizeRealSchedule_conv_PSB(input_dim, output_node, int_node):

    input1 = Input(shape=(input_dim,))
    print int_node
    x = Dense(int_node, activation='tanh')(input1)
    real_out = Dense(output_node, activation='linear')(x)

    return input1, real_out

def OptimizeBinSchedule_conv_PSB(input_dim, output_node, int_node):

    input2 = Input(shape=(input_dim,))
    print int_node
    x = Dense(int_node, activation='tanh')(input2)
    bin_out = Dense(output_node, activation='sigmoid')(x)

    return input2, bin_out


#Creating Real and Binary Schedule

def CreateRealSchedule_conv_PSB(input_dim, output_dim, int_node):

    input1 = Input(shape=(input_dim,))
    x = Dense(int_node, activation='sigmoid')(input1)
    real_out = Dense(output_dim, activation='linear')(x)

    return input1, real_out

def CreateBinSchedule_conv_PSB(input_dim, output_dim, int_node):
    # Bin Model
    input2 = Input(shape=(input_dim,))
    x = Dense(int_node, activation='tanh')(input2)
    bin_out = Dense(output_dim, activation='sigmoid')(x)

    return input2, bin_out



def NN_optimizeNN_v2(t, X, Y, params):
    row_max, col_max = t.shape

    t_cv1, real_cv1 = OptimizeRealSchedule_conv_PSB(col_max, params['real_num'], params['real_units'])
    t_cv2, bin_cv1 = OptimizeBinSchedule_conv_PSB(col_max, params['bin_num'], params['bin_units'])
    #weather_input = Input(shape=(weather_idxMax,), name='weather_input')
    x = merge([real_cv1, bin_cv1], mode='concat')
    #x = Dense(params['layer2_units'], activation='hard_sigmoid')(x)
    main_cv = Dense(1, activation='linear')(x)

    cv_model = Model(input=[t_cv1, t_cv2], output=main_cv)
    cv_model.compile(loss='mse', optimizer='adam')


    # cv error
    cvscores = []
    kfold = KFold(n=len(Y), n_folds=3)
    iter_max = 3

    for train_index, test_index in kfold:
        t_train, t_test = t[train_index], t[test_index]
        #x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        e_t = numpy.zeros((iter_max, 1))
        e_select = 1

        for i in range(0, iter_max):
            cv_model.fit([t_train, t_train], y_train, nb_epoch=50, batch_size=1, verbose=0)
            y_cv = numpy.squeeze(cv_model.predict([t_test, t_test]))
            e_t[i] = (MathFunctions.rms_flat(y_cv- numpy.squeeze(y_test)))/(MathFunctions.rms_flat(numpy.squeeze(y_test)))

            if i > 0:
                if e_t[i] < e_t[i-1]:
                    e_select = e_t[i]
            else:
                e_select = e_t[i]


        e_temp = e_select
        print e_select
        cvscores.append(e_temp)

    cvscores = numpy.asanyarray(cvscores)
    return cvscores


def fit_model(model, t, Y):
    # cv error

    kfold = KFold(n=len(Y), n_folds=3)
    iter_max = 5
    model_init = model
    save_model = model
    e_t = numpy.zeros((iter_max, 1))
    e_select = 1

    for i in range(0, iter_max):

        cvscores = []
        model = model_init

        for train_index, test_index in kfold:
            t_train, t_test = t[train_index], t[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            model.fit([t_train, t_train], y_train, nb_epoch=50, batch_size=20, verbose=0)
            y_cv = numpy.squeeze(model.predict([t_test, t_test]))
            e_temp = (MathFunctions.rms_flat(y_cv - numpy.squeeze(y_test)))/(MathFunctions.rms_flat(numpy.squeeze(y_test)))


            cvscores.append(e_temp)

        cvscores = numpy.asanyarray(cvscores)
        print cvscores
        e_t[i] = cvscores.mean()

        if i > 0:
            if e_t[i] < e_t[i - 1]:
                e_select = e_t[i]
                save_model = model
        else:
            e_select = e_t[i]

    print e_t
    print e_select

    return save_model, e_select


def PSB_model_DL(X_t, Y_t, X_e, best):
    binary_num = best['bin_num']
    bin_hyp = best['bin_units']

    real_num = best['real_num']
    real_hyp = best['real_units']

    # Fit model
    row_sch, col_max = X_t.shape
    t_input1, real_out = CreateRealSchedule_conv_PSB(col_max, real_num, real_hyp)
    t_input2, bin_out = CreateBinSchedule_conv_PSB(col_max, binary_num, bin_hyp)

    # merging models
    # weather_input
    # weather_input = Input(shape=(weather_idxMax,), name='weather_input')
    x = merge([real_out, bin_out], mode='concat')
    # x = Dense(layer2_hyp, activation='hard_sigmoid')(x)
    main_out = Dense(1, activation='linear')(x)

    main_model = Model(input=[t_input1, t_input2], output=main_out)
    main_model.compile(loss='mse', optimizer='adam')

    # Re-training model multiple times
    main_model, cvscores = fit_model(main_model, X_t, Y_t)

    main_model.fit([X_t, X_t], Y_t, nb_epoch=50, batch_size=20, verbose=0)
    Y_p = numpy.squeeze(main_model.predict([X_e, X_e]))

    return Y_p


def evaluate_performance(model, X, Y, scalar_var):
    trainScore = model.evaluate(X, Y, verbose=0)
    trainScore = math.sqrt(trainScore)
    trainScore = scalar_var.inverse_transform(numpy.array([[trainScore]]))

    return trainScore

def optimize_lstm_daily(X_day_t, H_day_t, space):

    def objective(params):
        optimize_model = build_lstm_v1.lstm_single_103(X_day_t.shape[2], H_day_t.shape[1], params)
        for ep in range(1):
            optimize_history = optimize_model.fit(X_day_t, H_day_t, batch_size=1, nb_epoch=1, validation_split=0.1, shuffle=False)
            optimize_model.reset_states()

        loss_v = optimize_history.history['loss']
        print loss_v

        loss_out = loss_v[-1]

        return {'loss': loss_out, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=3)

    return best

def fit_lstm_daily(X_day_t, H_day_t, best_params):
    lstm_hidden = best_params

    lstm_model = build_lstm_v1.lstm_single_103(X_day_t.shape[2], H_day_t.shape[1], lstm_hidden)

    for ep in range(50):
        lstm_history = lstm_model.fit(X_day_t, H_day_t, batch_size=1, nb_epoch=1, validation_split=0.2, shuffle=False)
        lstm_model.reset_states()

    lstm_history = lstm_model.fit(X_day_t, H_day_t, batch_size=1, nb_epoch=1, validation_split=0.05, shuffle=False)

    return lstm_model