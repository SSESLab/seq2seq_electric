from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Merge, Input
from keras.utils.visualize_util import plot
from keras.models import Model
from keras.layers import merge
from keras.callbacks import EarlyStopping
import numpy
import pydot
import keras
print pydot.find_graphviz()

from keras import backend as K

#Cross validation
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
import sys

#hyperas modules
import pymongo
import hyperas
import hyperopt
from hyperopt import Trials, STATUS_OK, tpe
from hyperopt import fmin, tpe, hp

#custom libraries
import MathFunctions


#Space for hyper parameters
space = {
         'units1': hp.uniform('units1', 64, 1024),
         'units2': hp.uniform('units2', 64, 1024),

         'dropout1': hp.uniform('dropout1', .25, .75),
         'dropout2': hp.uniform('dropout2', .25, .75),

         'batch_size': hp.uniform('batch_size', 28, 128),

         'nb_epochs': 100
         }

def CreateScheduleModel(dropout_rate=0.0):
    # Building a schedule model
    ScheduleModel = Sequential()
    ScheduleModel.add(Dense(20, input_dim=1))
    ScheduleModel.add(Activation('hard_sigmoid'))
    #ScheduleModel.add(Dropout(dropout_rate))
    ScheduleModel.add(Dense(1))
    ScheduleModel.add(Activation('linear'))

    # Compile model
    # keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    ScheduleModel.compile(loss='mse', optimizer='adam')
    plot(ScheduleModel, to_file='ScheduleModel2.png')
    return ScheduleModel

#Create a binary schedule
def CreateBinaryScheduleModel(dropout_rate=0.0):
    # Building a schedule model
    BinScheduleModel = Sequential()
    BinScheduleModel.add(Dense(20, input_dim=1))
    BinScheduleModel.add(Activation('tanh'))
    BinScheduleModel.add(Dropout(dropout_rate))
    BinScheduleModel.add(Dense(1))
    BinScheduleModel.add(Activation('sigmoid'))

    # Compile model
    # keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    BinScheduleModel.compile(loss='binary_crossentropy', optimizer='adadelta')
    plot(BinScheduleModel, to_file='BinScheduleModel2.png')
    return BinScheduleModel


def OptimizeScheduleModel(X_train, Y_train, X_test, Y_test):
    # Building a schedule model
    ScheduleModel = Sequential()
    ScheduleModel.add(Dense(20, input_dim=1))
    ScheduleModel.add(Activation('hard_sigmoid'))
    ScheduleModel.add(Dropout({{uniform(0, 1)}}))
    ScheduleModel.add(Dense(1))
    ScheduleModel.add(Activation('linear'))

    #compiling model
    ScheduleModel.compile(loss='mse', optimizer='adam')

    ScheduleModel.fit(X_train, Y_train, batch_size=20, nb_epoch=1, verbose=2, validation_data=(X_test, Y_test))

    #assessing score
    score = ScheduleModel.evaluate(X_test, Y_test, verbose=0)
    return {'loss': -score, 'status': STATUS_OK, 'model': ScheduleModel}

def CreateMainModel(dropout_rate=0.0):
    # Building the model for prediction model
    MainModel = Sequential()
    MainModel.add(Dense(20, input_dim=11))
    MainModel.add(Activation('sigmoid'))
    MainModel.add(Dropout(dropout_rate))
    MainModel.add(Dense(1))

    # Compile model
    MainModel.compile(loss='mse', optimizer='adam')
    plot(MainModel, to_file='MainModel2.png')
    return MainModel


def CreateBinSchedule_v2(x_dim, y_dim):
    inputs = Input(shape=(x_dim, ))

    x = Dense(512, activation='tanh')(inputs)
    predictions = Dense(y_dim, activation='sigmoid')(x)

    BinModel = Model(input=inputs, output=predictions)
    BinModel.compile(loss='binary_crossentropy', optimizer='adadelta')

    return BinModel, predictions


def CreateRealSchedule_v2(x_dim, y_dim):
    inputs = Input(shape=(x_dim, ))

    x = Dense(50, activation='hard_sigmoid')(inputs)
    #x = Dense(100, activation = 'sigmoid')(x)
    predictions = Dense(y_dim, activation='linear')(x)

    RealModel = Model(input=inputs, output=predictions)
    RealModel.compile(loss='mse', optimizer='adam')

    return RealModel, predictions

def CreateRealSchedule_DL(input_dim, output_dim, int_node):

    input1 = Input(shape=(input_dim,))
    x = Dense(int_node, activation='hard_sigmoid')(input1)
    real_out = Dense(output_dim, activation='linear')(x)

    return input1, real_out

def CreateBinSchedule_DL(input_dim, output_dim, int_node):
    # Bin Model
    input2 = Input(shape=(input_dim,))
    x = Dense((2**int_node), activation='tanh')(input2)
    bin_out = Dense(output_dim, activation='sigmoid')(x)

    return input2, bin_out

def OptimizeRealSchedule_DL(input_dim, output_node, int_node):

    input1 = Input(shape=(input_dim,))
    print int_node
    x = Dense(int_node, activation='hard_sigmoid')(input1)
    real_out = Dense(output_node, activation='linear')(x)

    return input1, real_out

def OptimizeBinSchedule_DL(input_dim, output_node, int_node):

    input2 = Input(shape=(input_dim,))
    print int_node
    x = Dense((2**int_node), activation='tanh')(input2)
    bin_out = Dense(output_node, activation='sigmoid')(x)

    return input2, bin_out

def NN_optimizeNN(t, X, Y, real_num, binary_num, weather_idxMax, params):
    t_cv1, real_cv1 = OptimizeRealSchedule_DL(1, real_num, params['real_units'])
    t_cv2, bin_cv1 = OptimizeBinSchedule_DL(1, binary_num, params['bin_units'])
    weather_input = Input(shape=(weather_idxMax,), name='weather_input')
    x = merge([real_cv1, bin_cv1, weather_input], mode='concat')
    x = Dense(params['layer2_units'], activation='hard_sigmoid')(x)
    main_cv = Dense(1, activation='linear')(x)

    cv_model = Model(input=[t_cv1, t_cv2, weather_input], output=main_cv)
    cv_model.compile(loss='mse', optimizer='adam')


    # cv error
    cvscores = []
    kfold = KFold(n=len(Y), n_folds=3)
    iter_max = 3

    for train_index, test_index in kfold:
        t_train, t_test = t[train_index], t[test_index]
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        e_t = numpy.zeros((iter_max, 1))
        e_select = 1

        for i in range(0, iter_max):
            cv_model.fit([t_train, t_train, x_train], y_train, nb_epoch=50, batch_size=1, verbose=0)
            y_cv = numpy.squeeze(cv_model.predict([t_test, t_test, x_test]))
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



def fit_model(model, t, X, Y):
    # cv error

    kfold = KFold(n=len(Y), n_folds=3)
    iter_max = 10
    model_init = model
    save_model = model
    e_t = numpy.zeros((iter_max, 1))
    e_select = 1

    for i in range(0, iter_max):

        cvscores = []
        model = model_init

        for train_index, test_index in kfold:
            t_train, t_test = t[train_index], t[test_index]
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            model.fit([t_train, t_train, x_train], y_train, nb_epoch=200, batch_size=20, verbose=0)
            y_cv = numpy.squeeze(model.predict([t_test, t_test, x_test]))
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




def NN_optimizeNN_v2(t, X, Y, real_num, binary_num, params):
    row_max, col_max = t.shape

    t_cv1, real_cv1 = OptimizeRealSchedule_DL(col_max, real_num, params['real_units'])
    t_cv2, bin_cv1 = OptimizeBinSchedule_DL(col_max, binary_num, params['bin_units'])
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


def CreateRealSchedule_v21(NN_hidden, x_dim):
    inputs = Input(shape=(x_dim, ))

    NN_h1 = int(NN_hidden['Layer1'])

    x = Dense(NN_h1, activation='relu')(inputs)
    #x = Dense(100, activation = 'sigmoid')(x)
    predictions = Dense(1, activation='linear')(x)

    RealModel = Model(input=inputs, output=predictions)
    RealModel.compile(loss='mean_squared_error', optimizer='adam')

    return RealModel

def CreateRealSchedule_v22(NN_hidden, x_dim):
    inputs = Input(shape=(x_dim, ))

    NN_h1 = int(NN_hidden['Layer1'])

    x = Dense(NN_h1, activation='tanh')(inputs)
    #x = Dense(100, activation = 'sigmoid')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    RealModel = Model(input=inputs, output=predictions)
    RealModel.compile(loss='mean_squared_error', optimizer='adam')

    return RealModel

def NN_optimizeNN_v21(X_t, Y_t, X_v, Y_v):
    space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        #'Layer2': hp.quniform('Layer2', 10, 100, 5),
        #'Layer3': hp.quniform('Layer3', 5, 20, 1),
        # 'D1': hp.uniform('D1', 0, 0.5),
        # 'D2': hp.uniform('D2', 0, 0.5),
        # 'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
    }

    def objective(params):
        # optimize_model = build_lstm_v1.lstm_model_102(params, train_data.shape[2], 24, 24)
        # optimize_model = build_lstm_v1.lstm_model_106(params, train_data.shape[2], 24)
        optimize_model = CreateRealSchedule_v21(params, X_t.shape[1])

        # for epochs in range(5):
        for ep in range(5):
            # optimize_history = optimize_model.fit(X_seq, Y_seq, batch_size=1, nb_epoch=3, validation_split=(X_seq, Y_seq), shuffle=False)
            optimize_history = optimize_model.fit(X_t, Y_t, batch_size=1, nb_epoch=1,
                                                  validation_data=(X_v, Y_v), shuffle=False)
            # optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.3, shuffle=False)
            #optimize_model.reset_states()

        loss_v = optimize_history.history['val_loss']
        print loss_v

        loss_out = loss_v[-1]

        return {'loss': loss_out, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=10)

    return best


def NN_optimizeNN_v22(X_t, Y_t, X_v, Y_v):
    space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        #'Layer2': hp.quniform('Layer2', 10, 100, 5),
        #'Layer3': hp.quniform('Layer3', 5, 20, 1),
        # 'D1': hp.uniform('D1', 0, 0.5),
        # 'D2': hp.uniform('D2', 0, 0.5),
        # 'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
    }

    def objective(params):
        # optimize_model = build_lstm_v1.lstm_model_102(params, train_data.shape[2], 24, 24)
        # optimize_model = build_lstm_v1.lstm_model_106(params, train_data.shape[2], 24)
        optimize_model = CreateRealSchedule_v22(params, X_t.shape[1])

        # for epochs in range(5):
        for ep in range(5):
            # optimize_history = optimize_model.fit(X_seq, Y_seq, batch_size=1, nb_epoch=3, validation_split=(X_seq, Y_seq), shuffle=False)
            optimize_history = optimize_model.fit(X_t, Y_t, batch_size=1, nb_epoch=1,
                                                  validation_data=(X_v, Y_v), shuffle=False)
            # optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0.3, shuffle=False)
            #optimize_model.reset_states()

        loss_v = optimize_history.history['val_loss']
        print loss_v

        loss_out = loss_v[-1]

        return {'loss': loss_out, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=10)

    return best



def model_optimizer_101(optimize_model, train_data, H_t, val_data, H_val, epoch_max):


    #for epochs in range(5):
    for ep in range(epoch_max):
        optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_data=(val_data, H_val), shuffle=False)
        optimize_model.reset_states()
#        get_3rd_layer_output = K.function([optimize_model.layers[0].input], [optimize_model.layers[6].output])
        #get_3rd_layer_output = K.function([optimize_model.layers[0].input], [optimize_model.layers[4].get_output_at(0)])
     #   X_temp = train_data[0, :, :]
      #  X_temp = X_temp[None, :, :]
       # layer_output = numpy.asarray(get_3rd_layer_output([X_temp]))
        #print layer_output
        #print layer_output.shape
        #print H_t[0]

    loss_v = optimize_history.history['val_loss']
    print loss_v

    loss_out = loss_v[-1]

    return loss_out



#def model_fit_101(lstm_model, train_data, H_t, val_data, H_val, e_mat, params):





    #return None



def param_initialize_101(attempt_max, epoch_max, min_epoch, tau, tol):
    # parameters for simulation

    # Criterion for early stopping

    e_mat = numpy.zeros((epoch_max, attempt_max))
    e_temp = numpy.zeros((tau,))

    count = 0
    val_loss_v = []
    epsilon = 1  # initialzing error
    loss_old = 1
    loss_val = 1

    param_list = [epoch_max, min_epoch, tau, tol, count, epsilon, loss_old, loss_val]

    return e_mat, e_temp, param_list, val_loss_v



