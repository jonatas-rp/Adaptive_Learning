import os
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from src import data_processor


class Experiment:

    def __init__(self, symbol, data, neurons, epochs):
        self.symbol = symbol
        self.data = data
        self.neurons = neurons
        self.epochs = epochs

    def create_model(self, neurons, window_size):    
        model = Sequential()
        model.add(Dense(neurons, input_dim=window_size, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def check_model(self, fpath, neurons, window_size):
        # Check if model exists
        if os.path.isfile(fpath + 'btc_checkpoint.h5'):
            model = load_model(fpath + 'btc_checkpoint.h5')
        else:
            model = self.create_model(neurons, window_size)
        return model

    def run_experiment(self):

        for n in self.neurons:
            for window_size in range(1, 365):
                # Initializing
                i = 0
                dataset = self.data[:window_size]
                history_loss = []

                singular_rmse = list()
                compound_rmse = list()

                # Callbacks
                fpath = './tests/'+ self.symbol +'/noincrement_es_tests/neurons_'+str(n)+'/window_size_' + \
                    str(window_size) + '/'

                model = self.check_model(fpath, n, window_size)

                # Check if predictions exists
                if os.path.isfile('predictions.csv'):
                    predictions = pd.read_csv('predictions.csv', sep=';', index_col=0)
                    predictions = predictions.reset_index(drop=True)
                else:
                    predictions = pd.DataFrame(columns=['Predictions'])
                    predictions_test = pd.DataFrame(columns=['Predictions Test'])

                # Iterate over all database
                while len(dataset) < len(self.data):
                    i = i + 1
                    print('%d neurons; %d window_size; %d amostras' %
                        (n, window_size, i))

                    # Add a new sample
                    dataset = data_processor.add_sample(self.data, dataset)

                    # Get training and test set
                    X_train, Y_train = data_processor.get_train_data(dataset, window_size)
                    
                    X_test, Y_test = data_processor.get_test_data(dataset, window_size)
        
                    model.fit(X_train, Y_train, epochs=self.epochs,verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=1)])
                    history_loss = history_loss + model.history.history['loss']

                    prediction = model.predict(X_test)

                    predictions = predictions.append(pd.DataFrame(prediction, columns=['Predictions'])).reset_index(drop=True)

                    singular_rmse.append(math.sqrt(mean_squared_error(Y_test, prediction)))
                    compound_rmse.append(math.sqrt(mean_squared_error(dataset[window_size:], predictions)))

                # Save NN model
                model.save(fpath + 'btc_checkpoint.h5')

                compare = data_processor.get_compare(self.data, dataset, predictions, singular_rmse, compound_rmse, window_size)

                # Check if folder exists
                if not os.path.exists(fpath):
                    os.mkdir(fpath)

                data_processor.save_csv(compare, history_loss, window_size, fpath)
