import keras

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Reshape
from keras.layers import Input, RepeatVector
from keras.layers import merge
from keras import backend as K
from keras.layers.core import ActivityRegularization

#testing backend


from keras.regularizers import l2
from keras.regularizers import activity_l1

from keras.utils.visualize_util import plot
from keras.utils.layer_utils import layer_from_config

import numpy

def lstm_model_101(lstm_hidden, time_lag, X, Y):
    model = Sequential()
    model.add(LSTM(lstm_hidden, input_dim=1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, nb_epoch=100, batch_size=1, verbose=2)

    return model

def lstm_stateful_101(lstm_hidden, X, Y):
    batch_size = 1
    model = Sequential()
    model.add(LSTM(lstm_hidden, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(Y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(300):
        model.fit(X, Y, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False) #the whole batch is encoded in time steps
        model.reset_states()

    return model

def lstm_model_102(lstm_hidden, in_dim, out_dim, tsteps):
    #Extracting hidden layers
    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])

    lstm_d1 = (lstm_hidden['D1'])
    lstm_d2 = (lstm_hidden['D2'])

    batch_size = 1
    model = Sequential()
    model.add(LSTM(lstm_h1, return_sequences=True, batch_input_shape=(batch_size, tsteps, in_dim), stateful=True)) #outputs lstm_hidden
    model.add(Dropout(lstm_d1))
    model.add(LSTM(lstm_h2, return_sequences=False, forget_bias_init='one', stateful=True)) #accepts lstm_hidden
    model.add(Dropout(lstm_d2))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



def lstm_optimize_102(in_dim, out_dim, tsteps, params):
    #Extracting hidden layers
    lstm_h1 = int(params['Layer1'])
    lstm_h2 = int(params['Layer2'])

    #lstm_d1 = params['D1']
    #lstm_d2 = params['D2']

    batch_size = 1
    model = Sequential()
    model.add(LSTM(lstm_h1, return_sequences=True, batch_input_shape=(batch_size, tsteps, in_dim), stateful=True)) #outputs lstm_hidden
    model.add(Dropout(0.2))
    model.add(LSTM(lstm_h2, return_sequences=False, forget_bias_init='one', stateful=True)) #accepts lstm_hidden
    model.add(Dropout(0.2))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def lstm_single_103(in_dim, out_dim, params):

    lstm_h1 = int(params['Layer1'])

    batch_size = 1
    model = Sequential()
    model.add(LSTM(lstm_h1, batch_input_shape=(batch_size, 1, in_dim), stateful=True))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def lstm_model_104(lstm_hidden, in_dim, out_dim, tsteps):
    #Extracting hidden layers
    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])

    lstm_d1 = (lstm_hidden['D1'])
    lstm_d2 = (lstm_hidden['D2'])

    batch_size = 1
    model = Sequential()
    model.add(LSTM(lstm_h1, return_sequences=True, batch_input_shape=(batch_size, tsteps, in_dim), stateful=True)) #outputs lstm_hidden
    model.add(Dropout(lstm_d1))
    model.add(LSTM(lstm_h2, return_sequences=False, forget_bias_init='one', stateful=True)) #accepts lstm_hidden
    model.add(Dropout(lstm_d2))
    model.add(Dense(out_dim, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def lstm_model_105(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])

    #lstm_d1 = (lstm_hidden['D1'])
    #lstm_d2 = (lstm_hidden['D2'])

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input((tsteps, in_dim))
    encoder = LSTM(lstm_h1, return_sequences=False)(inputs)
    #Adding a dense layer
    encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True)(encoding_repeat)

    #decoder_2 = LSTM(24, return_sequences=False, activation='linear')(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2)
    sequence_prediction = TimeDistributed(Dense(1, activation='linear'))(decoder)

    #Now the 2_D part
    #x = merge([sequence_prediction, inputs], mode='concat', concat_axis=2)
    x = sequence_prediction
    x = Reshape((tsteps,))(x)
    x = (Dense(lstm_h3, activation='relu'))(x)
    out_3D = (Dense(tsteps, activation='linear'))(x)

    model  = Model(inputs, out_3D)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def lstm_model_106(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])

    #lstm_d1 = (lstm_hidden['D1'])
    #lstm_d2 = (lstm_hidden['D2'])

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, return_sequences=False, stateful=True)(inputs)

    #printing model
    #print encoder.get_config()
    #encoder = LSTM(lstm_h1, return_sequences=False, stateful=True, dropout_U=0.5)(inputs)
    #Adding a dense layer
    encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True, stateful=True)(encoding_repeat)

    #decoder_2 = LSTM(24, return_sequences=False, activation='linear')(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2)
    sequence_prediction = TimeDistributed(Dense(1, activation='linear'))(decoder)

    #Now the 2_D part
    #x = merge([sequence_prediction, inputs], mode='concat', concat_axis=2)
    x = sequence_prediction
    x = Reshape((tsteps,))(x)
    x = (Dense(lstm_h3, activation='relu'))(x)
    out_3D = (Dense(tsteps, activation='linear'))(x)

    model = Model(inputs, out_3D)

    model.compile(loss='mean_squared_error', optimizer='adam')

    plot(model, to_file='LSTM_model_106.png')

    print "Printing Layers"
    for i in range(0, 10):
        print model.layers[i].output_shape


    return model

def lstm_model_107(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])
    activ_l3 = lstm_hidden['activ_l3']

    #lstm_d1 = (lstm_hidden['D1'])
    #lstm_d2 = (lstm_hidden['D2'])

    batch_size = 1
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, return_sequences=False, stateful=True)(inputs)
    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='tanh')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True, stateful=True,  W_regularizer=l2(0.01), activation='tanh')(encoding_repeat)
    #decoder = ActivityRegularization(l1=0.01)(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2)
    sequence_prediction = TimeDistributed(Dense(1, activation='linear'))(decoder)

    #Now the 2_D part
    x = sequence_prediction
    x = Reshape((tsteps,))(x)
    x = (Dense(lstm_h3, activation=activ_l3))(x)
    out= (Dense(tsteps, activation='sigmoid'))(x)


    model = Model(inputs, out)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam')
    plot(model, to_file='LSTM_model_107.png')

    return model

def prepare_sequences(x_train, y_train, window_length):
    row_max, col_max = x_train.shape
    #x_train = numpy.reshape(x_train, (int(row_max/window_length), window_length, col_max))
    #y_train = numpy.reshape(y_train, (int(row_max/window_length), window_length, 1))

    windows = numpy.zeros((1, window_length, col_max))
    windows_y = numpy.zeros((1, window_length, 1))

    for window_start in range(0, row_max - window_length + 1):
        window_end = window_start + window_length
        window = x_train[window_start:window_end, :]
        window_y = y_train[window_start:window_end]
        window = numpy.reshape(window, (1, window_length, col_max))
        window_y = numpy.reshape(window_y, (1, window_length, 1))

        #print window
        #print windows_y

        windows = numpy.concatenate((windows, window), axis=0)
        windows_y = numpy.concatenate((windows_y, window_y), axis=0)

        #windows = numpy.delete(windows, (0), axis=0)
        #windows_y = numpy.delete(windows_y, (0), axis=0)

    return windows, windows_y

def lstm_model_108(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])
    activ_l3 = lstm_hidden['activ_l3']

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, return_sequences=False, activation='hard_sigmoid', stateful=True)(inputs)

    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True,  activation='hard_sigmoid',  W_regularizer=l2(0.01), stateful=True)(encoding_repeat)
    #decoder = ActivityRegularization(l1=0.01)(decoder)
    #decoder_2 = LSTM(24, return_sequences=False, activation='linear')(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2)
    sequence_prediction = TimeDistributed(Dense(1,  activation='linear'))(decoder)

    #Now the 2_D part
    #x = merge([sequence_prediction, inputs], mode='concat', concat_axis=2)
    x = sequence_prediction
    x = Reshape((tsteps,))(x)
    x = (Dense(lstm_h3,  activation='relu'))(x)
    out_3D = (Dense(tsteps, activation='linear'))(x)

    model = Model(inputs, out_3D)

    model.compile(loss='mean_squared_error', optimizer='adam')

    plot(model, to_file='LSTM_model_108.png')


    return model


def lstm_model_109(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])
    activ_l3 = lstm_hidden['activ_l3']

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, activation='hard_sigmoid', return_sequences=False, stateful=True)(inputs)

    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True, activation='hard_sigmoid', W_regularizer=l2(0.01), stateful=True)(encoding_repeat)
    #decoder = ActivityRegularization(l2=0.01)(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2) #THis is what I want to input
    #sequence_prediction = TimeDistributed(Dense(1,  activation='linear'))(decoder)


    #Creatinbg graph for Multi-layer perceptron
    input_MLP = Input(shape=(tsteps, lstm_h2+in_dim), batch_shape=(1, tsteps, lstm_h2+in_dim))
    x = TimeDistributed(Dense(lstm_h3,  activation=activ_l3))(input_MLP)
    predictions_MLP = TimeDistributed(Dense(1, activation='linear'))(x)
    model_MLP = Model(input=input_MLP, output=predictions_MLP)



    #Now the 2_D part
    predictions = model_MLP(decoder)
    out_3D = Reshape((tsteps, ))(predictions) #might have to reshape

    model = Model(inputs, out_3D)

    model.compile(loss='mean_squared_error', optimizer='adam')

    plot(model, to_file='LSTM_model_109.png')

    print "Printing Layers"
    for i in range(0, 7):
        print model.layers[i].output_shape


    return model


def lstm_model_109b(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])
    activ_l3 = lstm_hidden['activ_l3']

    print "Activation Layer is: "
    print activ_l3

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, return_sequences=False, activation='hard_sigmoid', stateful=True)(inputs)

    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True,  activation='hard_sigmoid', W_regularizer=l2(0.01), stateful=True)(encoding_repeat)
    #decoder = ActivityRegularization(l1=0.01)(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2) #THis is what I want to input


    #Creatinbg graph for Multi-layer perceptron
    input_MLP = Input(shape=(tsteps, lstm_h2+in_dim), batch_shape=(1, tsteps, lstm_h2+in_dim))
    x = TimeDistributed(Dense(lstm_h3,  activation=activ_l3))(input_MLP)
    predictions_MLP = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    model_MLP = Model(input=input_MLP, output=predictions_MLP)



    #Now the 2_D part
    predictions = model_MLP(decoder)
    out_3D = Reshape((tsteps, ))(predictions) #might have to reshape

    model = Model(inputs, out_3D)

    model.compile(loss='mean_squared_error', optimizer='adam')

    plot(model, to_file='LSTM_model_109b.png')


    return model



def GRU_model_101(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = GRU(lstm_h1, return_sequences=False,  stateful=True)(inputs)

    #Adding a dense layer
    encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = GRU(lstm_h2, return_sequences=True,  W_regularizer=l2(0.01), stateful=True)(encoding_repeat)
    #decoder = ActivityRegularization(l1=0.01)(decoder)
    #decoder_2 = LSTM(24, return_sequences=False, activation='linear')(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2)
    sequence_prediction = TimeDistributed(Dense(1,  activation='linear'))(decoder)

    #Now the 2_D part
    #x = merge([sequence_prediction, inputs], mode='concat', concat_axis=2)
    x = sequence_prediction
    x = Reshape((tsteps,))(x)
    x = (Dense(lstm_h3,  activation='relu'))(x)
    out_3D = (Dense(tsteps, activation='linear'))(x)

    model = Model(inputs, out_3D)

    model.compile(loss='mean_squared_error', optimizer='adam')

    print "Printing Layers"
    for i in range(0, 10):
        print model.layers[i].output_shape


    return model



def GRU_model_101b(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])

    #lstm_d1 = (lstm_hidden['D1'])
    #lstm_d2 = (lstm_hidden['D2'])

    batch_size = 1
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = GRU(lstm_h1, return_sequences=False, stateful=True)(inputs)
    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='tanh')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = GRU(lstm_h2, return_sequences=True, stateful=True, W_regularizer=l2(0.01), activation='tanh')(encoding_repeat)
    #decoder = ActivityRegularization(l1=0.01)(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2)
    sequence_prediction = TimeDistributed(Dense(1, activation='linear'))(decoder)

    #Now the 2_D part
    x = sequence_prediction
    x = Reshape((tsteps,))(x)
    x = (Dense(lstm_h3, activation='relu'))(x)
    out= (Dense(tsteps, activation='sigmoid'))(x)

    model = Model(inputs, out)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

#######


def GRU_model_102(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = GRU(lstm_h1, return_sequences=False, stateful=True, activation='hard_sigmoid')(inputs)

    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = GRU(lstm_h2, return_sequences=True, activation='hard_sigmoid', W_regularizer=l2(0.01), stateful=True)(encoding_repeat)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2) #THis is what I want to input

    #Creatinbg graph for Multi-layer perceptron
    input_MLP = Input(shape=(tsteps, lstm_h2+in_dim), batch_shape=(1, tsteps, lstm_h2+in_dim))
    x = TimeDistributed(Dense(lstm_h3,  activation='relu'))(input_MLP)
    predictions_MLP = TimeDistributed(Dense(1, activation='linear'))(x)
    model_MLP = Model(input=input_MLP, output=predictions_MLP)


    #Now the 2_D part
    predictions = model_MLP(decoder)
    out_3D = Reshape((tsteps, ))(predictions) #might have to reshape
    model = Model(inputs, out_3D)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print "Printing Layers"
    for i in range(0, 7):
        print model.layers[i].output_shape


    return model



#####


def GRU_model_102b(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])

    batch_size = 1
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = GRU(lstm_h1, return_sequences=False, stateful=True)(inputs)

    #Adding a dense layer
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = GRU(lstm_h2, return_sequences=True,  activation='tanh', W_regularizer=l2(0.00000001), stateful=True)(encoding_repeat)
    #decoder = ActivityRegularization(l1=0.01)(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2) #THis is what I want to input

    #Creatinbg graph for Multi-layer perceptron
    input_MLP = Input(shape=(tsteps, lstm_h2+in_dim), batch_shape=(1, tsteps, lstm_h2+in_dim))
    x = TimeDistributed(Dense(lstm_h3,  activation='relu'))(input_MLP)
    predictions_MLP = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    model_MLP = Model(input=input_MLP, output=predictions_MLP)

    #Now the 2_D part
    predictions = model_MLP(decoder)
    out_3D = Reshape((tsteps, ))(predictions) #might have to reshape
    model = Model(inputs, out_3D)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



def lstm_model_110(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])
    activ_l3 = lstm_hidden['activ_l3']
    activ_l4 = lstm_hidden['activ_l4']

    print activ_l3
    print activ_l4

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, activation='hard_sigmoid', return_sequences=False, stateful=True)(inputs)

    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True, activation='hard_sigmoid', W_regularizer=l2(0.0), stateful=True)(encoding_repeat)
    #decoder = ActivityRegularization(l2=0.01)(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2) #THis is what I want to input


    #Creatinbg graph for Multi-layer perceptron
    input_MLP = Input(shape=(tsteps, lstm_h2+in_dim), batch_shape=(1, tsteps, lstm_h2+in_dim))
    x = TimeDistributed(Dense(lstm_h3,  activation=activ_l3))(input_MLP)
    predictions_MLP = TimeDistributed(Dense(1, activation=activ_l4))(x)
    model_MLP = Model(input=input_MLP, output=predictions_MLP)


    #Now the 2_D part
    predictions = model_MLP(decoder)
    out_3D = Reshape((tsteps, ))(predictions) #might have to reshape
    model = Model(inputs, out_3D)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def lstm_model_110(lstm_hidden, in_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])
    activ_l3 = lstm_hidden['activ_l3']
    activ_l4 = lstm_hidden['activ_l4']

    print activ_l3
    print activ_l4

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    inputs = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, activation='tanh', return_sequences=False, stateful=True)(inputs)

    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True, activation='tanh', W_regularizer=l2(0.0), stateful=True)(encoding_repeat)
    #decoder = ActivityRegularization(l2=0.01)(decoder)
    decoder = merge([decoder, inputs], mode='concat', concat_axis=2) #THis is what I want to input


    #Creatinbg graph for Multi-layer perceptron
    input_MLP = Input(shape=(tsteps, lstm_h2+in_dim), batch_shape=(1, tsteps, lstm_h2+in_dim))
    x = TimeDistributed(Dense(lstm_h3,  activation=activ_l3))(input_MLP)
    predictions_MLP = TimeDistributed(Dense(1, activation=activ_l4))(x)
    model_MLP = Model(input=input_MLP, output=predictions_MLP)


    #Now the 2_D part
    predictions = model_MLP(decoder)
    out_3D = Reshape((tsteps, ))(predictions) #might have to reshape
    model = Model(inputs, out_3D)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



def lstm_multi_101(lstm_hidden, in_dim, h1_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])
    activ_l3 = lstm_hidden['activ_l3']
    activ_l4 = lstm_hidden['activ_l4']

    print activ_l3
    print activ_l4

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    input_1 = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, activation='tanh', return_sequences=False, stateful=True)(input_1)

    print h1_dim

    #merging the inputs with h1_dim
    h_inputs = Input(shape=(h1_dim, ), batch_shape=(1, h1_dim))
    encoder = merge([encoder, h_inputs], mode='concat', concat_axis=1)

    #Adding a dense layer
    #encoder = Dense(lstm_h1, activation='relu')(encoder)
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True, activation='tanh', W_regularizer=l2(0.0), stateful=True)(encoding_repeat)
    decoder = merge([decoder, input_1], mode='concat', concat_axis=2) #THis is what I want to input


    #Creatinbg graph for Multi-layer perceptron
    input_MLP = Input(shape=(tsteps, lstm_h2+in_dim), batch_shape=(1, tsteps, lstm_h2+in_dim))
    x = TimeDistributed(Dense(lstm_h3,  activation=activ_l3))(input_MLP)
    predictions_MLP = TimeDistributed(Dense(1, activation=activ_l4))(x)
    model_MLP = Model(input=input_MLP, output=predictions_MLP)


    #Now the 2_D part
    predictions = model_MLP(decoder)
    out_3D = Reshape((tsteps, ))(predictions) #might have to reshape
    model = Model(input=[input_1, h_inputs], output=out_3D)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print "Printing Layers"
    for i in range(0, 7):
        print model.layers[i].output_shape

    return model



def lstm_multi_101b(lstm_hidden, in_dim, h2_dim, tsteps):

    lstm_h1 = int(lstm_hidden['Layer1'])
    lstm_h2 = int(lstm_hidden['Layer2'])
    lstm_h3 = int(lstm_hidden['Layer3'])
    activ_l3 = lstm_hidden['activ_l3']
    activ_l4 = lstm_hidden['activ_l4']

    print activ_l3
    print activ_l4

    batch_size = 1
    #inputs = [None, tsteps, in_dim]
    input_1 = Input(shape=(tsteps, in_dim), batch_shape=(1, tsteps, in_dim))
    encoder = LSTM(lstm_h1, activation='tanh', return_sequences=False, stateful=True)(input_1)

    #merging the inputs with h2_dim
    h_inputs = Input(shape=(tsteps, h2_dim), batch_shape=(1, tsteps, h2_dim))


    #Adding a dense layer
    encoding_repeat = RepeatVector(tsteps)(encoder) #[None, in_dim] -> [None, tsteps, in_dim]
    decoder = LSTM(lstm_h2, return_sequences=True, activation='tanh', W_regularizer=l2(0.0), stateful=True)(encoding_repeat)
    decoder = merge([decoder, input_1, h_inputs], mode='concat', concat_axis=2) #THis is what I want to input


    #Creatinbg graph for Multi-layer perceptron
    input_MLP = Input(shape=(tsteps, lstm_h2+in_dim), batch_shape=(1, tsteps, lstm_h2+in_dim))
    x = TimeDistributed(Dense(lstm_h3,  activation=activ_l3))(input_MLP)
    predictions_MLP = TimeDistributed(Dense(1, activation='hard_sigmoid'))(x)
    model_MLP = Model(input=input_MLP, output=predictions_MLP)


    #Now the 2_D part
    predictions = model_MLP(decoder)
    out_3D = Reshape((tsteps, ))(predictions) #might have to reshape
    model = Model(input=[input_1, h_inputs], output=out_3D)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print "Printing Layers"
    for i in range(0, 7):
        print model.layers[i].output_shape

    return model


