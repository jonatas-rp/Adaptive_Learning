import os
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import numpy as np

from src import data_processor


class Experiment:

    def __init__(self, symbol, data, neurons, epochs = 100):
        self.symbol = symbol
        self.data = data
        self.neurons = neurons
        self.epochs = epochs

    def create_mlp_model(self, neurons, window_size):    
        model = Sequential()
        model.add(Dense(neurons, input_dim=window_size, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        return model


    def create_lstm_model(self, neurons, window_size):
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(1, 1, window_size), activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def check_model(self, fpath, neurons, window_size, model):
        # Check if model exists
        if os.path.isfile(fpath + self.symbol + '_checkpoint.h5'):
            model = load_model(fpath + self.symbol + '_checkpoint.h5')
        elif model == "mlp":
            model = self.create_mlp_model(neurons, window_size)
        elif model == "lstm":
            model = self.create_lstm_model(neurons, window_size)
        return model

    def fit_dataset(self, model, mtype, n, window_size, data, dataset, amostras=0):
        singular_rmse = list()
        compound_rmse = list()
        history_loss = []
        # Check if predictions exists
        if os.path.isfile('predictions.csv'):
            predictions = pd.read_csv('predictions.csv', sep=';', index_col=0)
            predictions = predictions.reset_index(drop=True)
        else:
            predictions = pd.DataFrame(columns=['Predictions'])

        # Iterate over all database
        while len(dataset) < len(data):
            amostras = amostras + 1
            print('%d neurons; %d window_size; %d amostras' %(n, window_size, amostras))

            # Add a new sample
            dataset = data_processor.add_sample(data, dataset)

            # Get training and test set
            X_train, Y_train = data_processor.get_train_data(dataset, window_size, mtype)    
            X_test, Y_test = data_processor.get_test_data(dataset, window_size, mtype)

            model.fit(X_train, Y_train, epochs=self.epochs,verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=1)])
            history_loss = history_loss + model.history.history['loss']

            prediction = model.predict(X_test)

            predictions = predictions.append(pd.DataFrame(prediction, columns=['Predictions'])).reset_index(drop=True)

            singular_rmse.append(math.sqrt(mean_squared_error(Y_test, prediction)))
            compound_rmse.append(math.sqrt(mean_squared_error(dataset[window_size:], predictions)))
        
        return history_loss, singular_rmse, compound_rmse, predictions

    def experiment(self, mtype, n, window_size):
        # Initializing
        dataset = self.data[:window_size]      

        # Callbacks
        fpath = './tests/'+ self.symbol +'/' + mtype + '_tests/neurons_'+str(n)+'/window_size_' + \
            str(window_size) + '/'

        model = self.check_model(fpath, n, window_size, mtype)

        #Train model
        history_loss, singular_rmse, compound_rmse, predictions = self.fit_dataset(model, mtype, n, window_size, self.data, dataset)

        # Save NN model
        model.save(fpath + self.symbol + '_checkpoint.h5')

        #Generate CSV
        compare = data_processor.get_compare(self.data, dataset, predictions, singular_rmse, compound_rmse, window_size)

        # Check if folder exists
        if not os.path.exists(fpath):
            os.mkdir(fpath)

        data_processor.save_csv(compare, history_loss, window_size, fpath)

    def run_experiment(self, mtype):
        for n in self.neurons:
            window_size = 1
            while window_size < 366:

                self.experiment(mtype, n, window_size)

                if window_size >= 90:
                    window_size += 3
                else:
                    window_size += 1

    def scaffold(self, mtype, window_size, k=4):
        nblocks = len(self.data)
        lendata = 0
        for i in range(nblocks):
            lendata += len(self.data[i])

        fpath = './tests/' + self.symbol + '/' + mtype + '_scaffold/neurons_' + str(self.neurons) + '/window_size_' + \
                str(window_size) + '/'


        model = self.check_model(fpath, self.neurons, window_size, mtype)
        data_test = pd.DataFrame(columns=['Data'])
        Y_test = pd.DataFrame(columns=['Y'])
        predictions_test = pd.DataFrame(columns=['Predictions'])
        singular_rmse_test = []
        compound_rmse_test = []

        #compare_test = pd.DataFrame(columns=['Data','Y','Predictions','Singular RMSE', 'Compound RMSE'])
        for i in range(0, k):
            amostras = 0

            print('%d neurons; %d window_size; %d repetitions;' % (self.neurons, window_size, i))
            #Train Phase
            dataBlock = data_processor.merge_blocks(self.data, i, nblocks - (k - i))
            dataset = dataBlock[:window_size]

            history_loss, singular_rmse, compound_rmse, predictions = self.fit_dataset(model, mtype, self.neurons, window_size, dataBlock, dataset)

            # Check if folder exists
            if not os.path.exists(fpath):
                os.mkdir(fpath)

            compare = data_processor.get_compare(dataBlock, dataBlock, predictions, singular_rmse, compound_rmse, window_size, nan=True)
            data_processor.save_csv(compare, history_loss, window_size, fpath, fname='compare_' + str(i) + '_')

            #Test Phase
            if window_size != 1:
                dataset = dataBlock[-(window_size-1):]

            dataBlock = self.data[nblocks - (k - i)]

            if window_size != 1:
                dataset = np.append(dataset, dataBlock, axis=0)
            else:
                dataset = dataBlock

            for j in range(len(dataBlock) - 1):
                # Get data
                X_test = dataset[j:window_size + j].reshape(-1, window_size)
                if mtype == 'lstm':
                    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                Y_test = Y_test.append(pd.DataFrame(dataBlock[j+1:j+2], columns=['Y'])).reset_index(drop=True)

                # Predict
                prediction_test = model.predict(X_test)
                predictions_test = predictions_test.append(
                    pd.DataFrame(prediction_test, columns=['Predictions'])).reset_index(drop=True)

                # Compute RMSE
                singular_rmse_test.append(
                    math.sqrt(mean_squared_error(dataBlock[j+1:j+2], prediction_test)))
                compound_rmse_test.append(
                    math.sqrt(mean_squared_error(Y_test, predictions_test)))

                # Retrain
                #model.fit(X_test, dataBlock[j+1:j+2], epochs=self.epochs,verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=0)])

            data_test = data_test.append(pd.DataFrame(dataBlock[:-1], columns=['Data'])).reset_index(drop=True)

            compare_test = data_processor.get_compare(data_test, Y_test, predictions_test, singular_rmse_test, compound_rmse_test, window_size)
            # Check if folder exists
            if not os.path.exists(fpath):
                os.mkdir(fpath)

        data_processor.save_csv(compare_test, history_loss, window_size, fpath, fname='noadpat')
        # Save NN model
        #model.save(fpath + self.symbol + '_checkpoint.h5')