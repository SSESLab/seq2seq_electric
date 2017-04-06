import numpy
import math
from scipy.interpolate import interp1d

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import fmin, tpe, hp
from keras.callbacks import EarlyStopping

#import custom libraries
import NNFun_PSB
import DataFunctions
import  MathFunctions
import build_lstm_v1

def train_test_split(X, Y):
    train_list = []
    test_list = []

    for i in range(0, len(Y)):
        if numpy.isnan(Y[i]):
            test_list.append(i)
        else:
            train_list.append(i)

    row_max, col_max = X.shape

    print "Length of test is:"
    print len(test_list)

    X_t = numpy.zeros((len(train_list), col_max))
    Y_t = numpy.zeros(len(train_list), )

    for i in range(0, len(train_list)):
        X_t[i, :] = X[train_list[i], :]
        Y_t[i] = Y[train_list[i]]

    X_e = numpy.zeros((len(test_list), col_max))
    Y_e = numpy.zeros(len(test_list), )

    for i in range(0, len(test_list)):
        X_e[i, :] = X[test_list[i], :]
        Y_e[i] = Y[test_list[i]]

    return X_t, Y_t, X_e, Y_e, train_list, test_list


def imputate_optimize(X_t, Y_t):

    #define space of hyperparameters
    space = {
        'bin_units': hp.quniform('bin_units', 1, 10, 1),
        'real_units': hp.quniform('real_units', 1, 10, 1),
        'real_num': hp.quniform('real_num', 1, 10, 1),
        'bin_num': hp.quniform('bin_num', 1, 10, 1),
        # 'layer2_units': hp.quniform('layer2_units', 10, 50, 1)
    }

    def objective(params):
        cvscores = NNFun_PSB.NN_optimizeNN_v2(X_t, 0, Y_t, params)
        e_temp = cvscores.mean()
        print e_temp
        return {'loss': e_temp, 'status': STATUS_OK}

    trials = Trials()

    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=1)

    return best


def organize_pred(Y, Y_p, test_list):


    for i in range(0, len(test_list)):
        Y[test_list[i]] = Y_p[i]

    return Y


def interpolate_main(X, Y, start_day, end_day, cons_points):

    #This code will do [1] create a corrupt list and [2]  return list for interp and LSTM


    Y = create_missing_value(Y, start_day, end_day)
    #p is just for verification

    small_list = []
    large_list = []

    i = 0

    #THIS BLOCK IS FOR SEGGREGATING SMALL LISTS FROM LARGE LISTS

    while i < len(Y):
        #find nan

        if numpy.isnan(Y[i]):

            if numpy.all(numpy.isnan(Y[i:i+cons_points])): #LARGE CHUNK CASE
                start_point = i
                glob_count = i + cons_points - 1

                while glob_count < len(Y) and numpy.isnan(Y[glob_count]):
                    glob_count = glob_count + 1

                i = glob_count
                end_point = glob_count
                large_list.append([start_point, end_point])

            else:  #Small LIST case
                small_list.append(i)

        i = i + 1

    #Let us start with the small list, i.e. interpolation case
    #Y_small_list = interp_linear(Y, small_list)
    #Y = Y_small_list[:, None]

    #Create a corrupt list here


    #Now we try to extract data for the data for LSTM values
    #Y_out = interp_LSTM(X, Y, large_list)



    return small_list, large_list



def interp_linear(Y, small_list):


    not_nan = numpy.logical_not(numpy.isnan(Y))
    not_nan = numpy.squeeze(not_nan)
    indices = numpy.arange(len(Y))
    indices = indices[:, None]


    f = interp1d(numpy.squeeze(indices[not_nan]), numpy.squeeze(Y[not_nan]))

    x_new = numpy.asarray(small_list)
    x_new = x_new[1:]
    #print x_new
    #print small_list

    y_lin = f(x_new)
    #print y_lin
    Y_out = numpy.squeeze(Y)
    Y_out[x_new] = y_lin
    Y_out[0] = Y_out[1]



    return Y_out



def interp_LSTM(X, Y, large_list):

    day_list = []
    Y_fp = Y.copy() #forward prediction
    Y_bp = Y.copy() #Backward prediction
    Y_out = numpy.squeeze(Y.copy())
    Y_mlp = numpy.squeeze(Y.copy())

    #first step,, select training and test data for each data set
    for a,b in large_list:
        start_point = math.trunc(a/24)
        end_point  = math.trunc(b/24)+1
        day_list.append([start_point, end_point])


    print day_list
    loop_count = 0

    for a,b in day_list:

        p = 0

        X1= numpy.empty((0, X.shape[1])) #X1 for before, X2 for after
        Y1 = numpy.empty((0, Y.shape[1]))
        X2 = numpy.empty((0, X.shape[1]))
        Y2 = numpy.empty((0, Y.shape[1]))

        if loop_count==0:
            X_add = X[0:a*24, :]
            Y_add = Y[0:a*24, :]
            X1 = numpy.append(X1, X_add, axis=0)
            Y1 = numpy.append(Y1, Y_add, axis=0)


            for i in range(loop_count+1,len(day_list)):
                c1, c2 = day_list[i-1]
                d1, d2 = day_list[i]
                X_a = X[c2*24:d1*24, :]
                Y_a = Y[c2*24:d1*24, :]
                X2 = numpy.append(X2, X_a, axis=0)
                Y2 = numpy.append(Y2, Y_a, axis=0)

                #This all matches up

            #Last Block

            X_add = X[d2*24:, :]
            Y_add = Y[d2*24:, :]
            X2 = numpy.append(X2, X_add, axis=0)
            Y2 = numpy.append(Y2, Y_add, axis=0)


        elif loop_count==len(day_list)-1:

            #Get the block before
            c1, c2 = day_list[0]
            X_add = X[0:c1*24, :]
            Y_add = Y[0:c1*24, :]
            X1 = numpy.append(X1, X_add, axis=0)
            Y1 = numpy.append(Y1, Y_add, axis=0)

            #MISSING ENDS!!!!c1
            for i in range(0, loop_count):
                c1, c2 = day_list[i]
                d1, d2 = day_list[i+1]
                X_a = X[c2 * 24:d1 * 24, :]
                Y_a = Y[c2 * 24:d1 * 24, :]
                X1 = numpy.append(X1, X_a, axis=0)
                Y1 = numpy.append(Y1, Y_a, axis=0)

            X_add = X[d2 * 24:, :]
            Y_add = Y[d2 * 24:, :]
            X2 = numpy.append(X2, X_add, axis=0)
            Y2 = numpy.append(Y2, Y_add, axis=0)


        else:
            # Get the block before
            c1, c2 = day_list[0]
            X_add = X[0:c1 * 24, :]
            Y_add = Y[0:c1 * 24, :]
            X1 = numpy.append(X1, X_add, axis=0)
            Y1 = numpy.append(Y1, Y_add, axis=0)

            # MISSING ENDS!!!!c1
            for i in range(0, loop_count):
                c1, c2 = day_list[i]
                d1, d2 = day_list[i + 1]
                X_a = X[c2 * 24:d1 * 24, :]
                Y_a = Y[c2 * 24:d1 * 24, :]
                X1 = numpy.append(X1, X_a, axis=0)
                Y1 = numpy.append(Y1, Y_a, axis=0)

            for i in range(loop_count + 1, len(day_list)):
                c1, c2 = day_list[i - 1]
                d1, d2 = day_list[i]
                X_a = X[c2 * 24:d1 * 24, :]
                Y_a = Y[c2 * 24:d1 * 24, :]
                X2 = numpy.append(X2, X_a, axis=0)
                Y2 = numpy.append(Y2, Y_a, axis=0)

            X_add = X[d2 * 24:, :]
            Y_add = Y[d2 * 24:, :]
            X2 = numpy.append(X2, X_add, axis=0)
            Y2 = numpy.append(Y2, Y_add, axis=0)




        #Extracting test values
        X_e = X[a*24:b*24, :]
        Y_e = Y[a*24:b*24, :]

        #Copying for NN
        X1_nn = X1.copy()
        Y1_nn = Y1.copy()
        X2_nn = X2.copy()
        Y2_nn = Y2.copy()
        X_e_nn = X_e.copy()



        #Y_e is ok here

        # Reshaping array into (#of days, 24-hour timesteps, #features)
        X1, Y1 = process_XY(X1, Y1, 24)
        X2, Y2 = process_XY(X2, Y2, 24)
        X_e, Y_e = process_XY(X_e, Y_e, 24)


        #Finding optimal parameters
        best = optimize_LSTM(X1, Y1)

        Y_p1 = LSTM_main(X1, Y1, X_e, best, 24)
        Y_p1 = Y_p1.flatten()
        #Y_p1= Y_p1[:, None]


        #reverse order predictions
        X_r = X2[::-1, :, :]
        Y_r = Y2[::-1, :]
        X_e2 = X_e[::-1, :, :]

        best = optimize_LSTM(X_r, Y_r)
        Y_p2 = LSTM_main(X_r, Y_r, X_e2, best, 24)
        Y_p2 = Y_p2[::-1, :]
        #print Y1[-1, :]



        #Fixing Y_out
        #THIS IS WHERE Y_E is going wrong
        Y_p2 = Y_p2.flatten()
        #Y_p2 = Y_p2[:, None]


        print Y_p1
        print Y_p2
        q_1 = float(len(X1))/float(len(X1) + len(X2))
        q_2 = float(len(X2))/float(len(X1) + len(X2))
        print q_1
        print q_2
        Y_out[a * 24:b * 24] = (q_1*Y_p1) + (q_2*Y_p2)
        print Y_p1.shape
        print Y_p2.shape

        print "Y_outputs"
        print Y_out[a * 24:b * 24]


        loop_count = loop_count + 1



        #block for MLP
        X_t = numpy.concatenate((X1_nn, X2_nn), axis=0)
        Y_t = numpy.concatenate((Y1_nn, Y2_nn), axis=0)
        Y_nn = interpolate_MLP(X_t, Y_t, X_e_nn)

        Y_mlp[a*24:b*24] = Y_nn



    return Y_out, Y_mlp




def interp_LSTM_v2(X, Y, large_list):

    day_list = []
    Y_out = numpy.squeeze(Y.copy())
    Y_mlp = numpy.squeeze(Y.copy())

    #first step,, select training and test data for each data set
    for a,b in large_list:
        start_point = math.trunc(a/24)
        end_point  = math.trunc(b/24)+1
        day_list.append([start_point, end_point])


    print day_list
    loop_count = 0

    for a,b in day_list:

        p = 0

        X1= numpy.empty((0, X.shape[1])) #X1 for before, X2 for after
        Y1 = numpy.empty((0, Y.shape[1]))
        X2 = numpy.empty((0, X.shape[1]))
        Y2 = numpy.empty((0, Y.shape[1]))

        if loop_count==0:
            X_add = X[0:a*24, :]
            Y_add = Y[0:a*24, :]
            X1 = numpy.append(X1, X_add, axis=0)
            Y1 = numpy.append(Y1, Y_add, axis=0)


            for i in range(loop_count+1,len(day_list)):
                c1, c2 = day_list[i-1]
                d1, d2 = day_list[i]
                X_a = X[c2*24:d1*24, :]
                Y_a = Y[c2*24:d1*24, :]
                X2 = numpy.append(X2, X_a, axis=0)
                Y2 = numpy.append(Y2, Y_a, axis=0)

                #This all matches up

            #Last Block

            X_add = X[d2*24:, :]
            Y_add = Y[d2*24:, :]
            X2 = numpy.append(X2, X_add, axis=0)
            Y2 = numpy.append(Y2, Y_add, axis=0)


        elif loop_count==len(day_list)-1:

            #Get the block before
            c1, c2 = day_list[0]
            X_add = X[0:c1*24, :]
            Y_add = Y[0:c1*24, :]
            X1 = numpy.append(X1, X_add, axis=0)
            Y1 = numpy.append(Y1, Y_add, axis=0)

            #MISSING ENDS!!!!c1
            for i in range(0, loop_count):
                c1, c2 = day_list[i]
                d1, d2 = day_list[i+1]
                X_a = X[c2 * 24:d1 * 24, :]
                Y_a = Y[c2 * 24:d1 * 24, :]
                X1 = numpy.append(X1, X_a, axis=0)
                Y1 = numpy.append(Y1, Y_a, axis=0)

            X_add = X[d2 * 24:, :]
            Y_add = Y[d2 * 24:, :]
            X2 = numpy.append(X2, X_add, axis=0)
            Y2 = numpy.append(Y2, Y_add, axis=0)


        else:
            # Get the block before
            c1, c2 = day_list[0]
            X_add = X[0:c1 * 24, :]
            Y_add = Y[0:c1 * 24, :]
            X1 = numpy.append(X1, X_add, axis=0)
            Y1 = numpy.append(Y1, Y_add, axis=0)

            # MISSING ENDS!!!!c1
            for i in range(0, loop_count):
                c1, c2 = day_list[i]
                d1, d2 = day_list[i + 1]
                X_a = X[c2 * 24:d1 * 24, :]
                Y_a = Y[c2 * 24:d1 * 24, :]
                X1 = numpy.append(X1, X_a, axis=0)
                Y1 = numpy.append(Y1, Y_a, axis=0)

            for i in range(loop_count + 1, len(day_list)):
                c1, c2 = day_list[i - 1]
                d1, d2 = day_list[i]
                X_a = X[c2 * 24:d1 * 24, :]
                Y_a = Y[c2 * 24:d1 * 24, :]
                X2 = numpy.append(X2, X_a, axis=0)
                Y2 = numpy.append(Y2, Y_a, axis=0)

            X_add = X[d2 * 24:, :]
            Y_add = Y[d2 * 24:, :]
            X2 = numpy.append(X2, X_add, axis=0)
            Y2 = numpy.append(Y2, Y_add, axis=0)




        #Extracting test values
        X_e = X[a*24:b*24, :]
        Y_e = Y[a*24:b*24, :]

        #Copying for NN
        X1_nn = X1.copy()
        Y1_nn = Y1.copy()
        X2_nn = X2.copy()
        Y2_nn = Y2.copy()
        X_e_nn = X_e.copy()



        #Y_e is ok here

        # Reshaping array into (#of days, 24-hour timesteps, #features)
        X1, Y1 = process_XY(X1, Y1, 24)
        X2, Y2 = process_XY(X2, Y2, 24)
        X_e, Y_e = process_XY(X_e, Y_e, 24)


        #Finding optimal parameters
        best = optimize_LSTM_v2(X1, Y1)

        Y_p1 = LSTM_main_v2(X1, Y1, X_e, best, 24)
        Y_p1 = Y_p1.flatten()
        #Y_p1= Y_p1[:, None]


        #reverse order predictions
        X_r = X2[::-1, :, :]
        Y_r = Y2[::-1, :]
        X_e2 = X_e[::-1, :, :]

        best = optimize_LSTM_v2(X_r, Y_r)
        Y_p2 = LSTM_main_v2(X_r, Y_r, X_e2, best, 24)
        Y_p2 = Y_p2[::-1, :]
        #print Y1[-1, :]



        #Fixing Y_out
        #THIS IS WHERE Y_E is going wrong
        Y_p2 = Y_p2.flatten()
        #Y_p2 = Y_p2[:, None]


        print Y_p1
        print Y_p2
        q_1 = float(len(X1))/float(len(X1) + len(X2))
        q_2 = float(len(X2))/float(len(X1) + len(X2))
        print q_1
        print q_2
        Y_out[a * 24:b * 24] = (q_1*Y_p1) + (q_2*Y_p2)
        print Y_p1.shape
        print Y_p2.shape

        print "Y_outputs"
        print Y_out[a * 24:b * 24]


        loop_count = loop_count + 1



        #block for MLP
        X_t = numpy.concatenate((X1_nn, X2_nn), axis=0)
        Y_t = numpy.concatenate((Y1_nn, Y2_nn), axis=0)
        Y_nn = interpolate_MLP(X_t, Y_t, X_e_nn)

        Y_mlp[a*24:b*24] = Y_nn



    return Y_out, Y_mlp





def process_XY(X, Y, intv):

    #normalize Y
    Y_rms = MathFunctions.rms_flat(Y)
    # normalizing data
    #H_min, H_max = DataFunctions.get_normalize_params(Y)
    #Y= Y / H_max

    #Reshape X, Y Pair
    row_seq = int(len(X)/intv)

    # Reshaping array into (#of days, 24-hour timesteps, #features)
    train_data = numpy.reshape(X, (row_seq, intv, X.shape[1]))
    H_out = numpy.reshape(Y, (row_seq, intv))


    return train_data, H_out



def optimize_LSTM(X, Y):

    numpy.random.seed(7)
    # This block is for optimizing LSTM layers
    space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 20, 1)
    }

    def objective(params):
        optimize_model = build_lstm_v1.lstm_model_106(params, X.shape[2], 24)

        # for epochs in range(5):
        for ep in range(5):
            optimize_history = optimize_model.fit(X, Y, batch_size=1, nb_epoch=1, validation_split=0.1, shuffle=False, verbose=0)
            optimize_model.reset_states()

        loss_v = optimize_history.history['val_loss']
        print loss_v

        loss_out = loss_v[-1]

        return {'loss': loss_out, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=10)

    return best


def optimize_LSTM_v2(X, Y):

    numpy.random.seed(7)
    # This block is for optimizing LSTM layers
    space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 20, 1)
    }

    def objective(params):
        optimize_model = build_lstm_v1.lstm_model_109b(params, X.shape[2], 24)

        # for epochs in range(5):
        for ep in range(5):
            optimize_history = optimize_model.fit(X, Y, batch_size=1, nb_epoch=1, validation_split=0.1, shuffle=False, verbose=0)
            optimize_model.reset_states()

        loss_v = optimize_history.history['val_loss']
        print loss_v

        loss_out = loss_v[-1]

        return {'loss': loss_out, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=10)

    return best



def LSTM_main(X, Y, X_e, best, intv):

    numpy.random.seed(7)
    lstm_hidden = best
    tsteps = intv
    #out_dim = intv

    lstm_model = build_lstm_v1.lstm_model_106(lstm_hidden, X.shape[2], tsteps)
    save_model = lstm_model

    ##callbacks for Early Stopping
    #callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    # parameters for simulation
    attempt_max = 5
    epoch_max = 100
    min_epoch = 10

    # Criterion for early stopping
    tau = 5
    e_mat = numpy.zeros((epoch_max, attempt_max))
    e_temp = numpy.zeros((tau,))

    tol = 0
    count = 0
    val_loss_v = []
    epsilon = 1  # initialzing error

    for attempts in range(attempt_max):
        lstm_model = build_lstm_v1.lstm_model_106(lstm_hidden, X.shape[2], tsteps)
        print "New model Initialized"

        for ep in range(epoch_max):
            lstm_history = lstm_model.fit(X, Y, batch_size=1, nb_epoch=1, validation_split=0.005, shuffle=False, verbose=0)

            loss_v = lstm_history.history['val_loss']
            val_loss_check = loss_v[-1]
            val_loss_v.append(val_loss_check)
            e_mat[ep, attempts] = val_loss_check

            # testing alternative block
            # lstm_model.reset_states()
            #y_val = lstm_model.predict(val_data, batch_size=1, verbose=0)
            #e1, e2 = DataFunctions.find_error(H_t, H_val, y_val)
            #print e1, e2
            #val_loss_check = e2
            #val_loss_v.append(val_loss_check)
            #e_mat[ep, attempts] = val_loss_check

            if val_loss_v[count] < epsilon:
                epsilon = val_loss_v[count]
                save_model = lstm_model
                #test_model = lstm_model
                #Y_lstm = lstm_model.predict(test_data, batch_size=1, verbose=0)
                #e_1, e_2 = DataFunctions.find_error(H_t, H_e, Y_lstm)
                #lstm_model.reset_states()
                #print e_1
                #print e_2

            count = count + 1
            lstm_model.reset_states()

            # This block is for early stopping
            if ep >= min_epoch:
                e_temp = e_mat[ep - tau + 1: ep + 1, attempts]
                e_local = e_mat[ep - tau, attempts]

                # print e_temp
                # print e_local

                if numpy.all(e_temp >= e_local):
                    break

    Y_lstm = save_model.predict(X_e, batch_size=1, verbose=0)



    return Y_lstm






def LSTM_main_v2(X, Y, X_e, best, intv):

    numpy.random.seed(7)
    lstm_hidden = best
    tsteps = intv
    #out_dim = intv

    lstm_model = build_lstm_v1.lstm_model_109b(lstm_hidden, X.shape[2], tsteps)
    save_model = lstm_model

    # parameters for simulation
    attempt_max = 5
    epoch_max = 100
    min_epoch = 10

    # Criterion for early stopping
    tau = 5
    e_mat = numpy.zeros((epoch_max, attempt_max))
    e_temp = numpy.zeros((tau,))

    tol = 0
    count = 0
    val_loss_v = []
    epsilon = 1  # initialzing error

    for attempts in range(attempt_max):
        lstm_model = build_lstm_v1.lstm_model_109b(lstm_hidden, X.shape[2], tsteps)
        print "New model Initialized"

        for ep in range(epoch_max):
            lstm_history = lstm_model.fit(X, Y, batch_size=1, nb_epoch=1, validation_split=0.005, shuffle=False, verbose=0)

            loss_v = lstm_history.history['val_loss']
            val_loss_check = loss_v[-1]
            val_loss_v.append(val_loss_check)
            e_mat[ep, attempts] = val_loss_check



            if val_loss_v[count] < epsilon:
                epsilon = val_loss_v[count]
                save_model = lstm_model


            count = count + 1
            lstm_model.reset_states()

            # This block is for early stopping
            if ep >= min_epoch:
                e_temp = e_mat[ep - tau + 1: ep + 1, attempts]
                e_local = e_mat[ep - tau, attempts]

                if numpy.all(e_temp >= e_local):
                    break

    Y_lstm = save_model.predict(X_e, batch_size=1, verbose=0)



    return Y_lstm










def create_missing_value(Y, start_day, end_day):

    Y_new = Y
    Y_new[start_day*24:(end_day*24-1), :] = numpy.nan

    return Y_new


def get_small_list(Y, small_list):
    i = 0

    while i<len(Y):
        if numpy.isnan(Y[i]):
            small_list.append(i)

        i = i +1

    return small_list


def interpolate_calculate_rms(Y_a, Y_p):

    not_nan = numpy.logical_not(numpy.isnan(Y_a))
    not_nan = numpy.squeeze(not_nan)

    A = Y_a[not_nan]
    B = Y_p[not_nan]

    e_interp = (MathFunctions.rms_flat(A - B)) / (MathFunctions.rms_flat(A))

    return e_interp



def interpolate_MLP(X, Y, X_e):

    best = imputate_optimize(X, Y)
    Y_p = NNFun_PSB.PSB_model_DL(X, Y, X_e, best)

    return Y_p