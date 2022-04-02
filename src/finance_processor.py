import math
from keras.models import load_model
from keras.callbacks import EarlyStopping

from src import data_processor
import numpy as np

class Finance:

    def __init__(self, symbol, data, neurons, window_size, epochs = 100, k=4):
        self.symbol = symbol
        self.data = data
        self.neurons = neurons
        self.window_size = window_size
        self.epochs = epochs
        self.k = k
        self.blocks = data_processor.split_by_year(self.data)
        self.model = None
        self.predictions = [[0]]

    def prepare_model(self, fpath, mtype = "lstm"):
        self.model = load_model(fpath + self.symbol + '_checkpoint.h5')

        dataTrain = data_processor.merge_blocks(self.blocks, 0, len(self.blocks)-self.k)
        dataTrain = np.append(dataTrain, [self.blocks[len(self.blocks)-self.k][0]], axis=0)
        dataset = dataTrain[:self.window_size]

        print("Training model....")
        while len(dataset) < len(dataTrain):

            dataset = data_processor.add_sample(dataTrain, dataset)

            X_train, Y_train = data_processor.get_train_data(dataset, self.window_size, mtype)

            self.model.fit(X_train, Y_train, epochs=self.epochs, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=0)])

    def forecaster(self, mtype='lstm'):
        dataTest = data_processor.merge_blocks(self.blocks, len(self.blocks) - self.k, len(self.blocks))
        dataTest = np.append(self.blocks[len(self.blocks) - self.k - 1][-(self.window_size-1):], dataTest, axis=0)

        dataset = dataTest[:self.window_size]

        while len(dataset) < len(dataTest):
            dataset = data_processor.add_sample(dataTest, dataset)

            X_test, _ = data_processor.get_train_data(dataset, self.window_size, mtype)

            # Forecast next day close
            prediction = self.model.predict(X_test)
            self.predictions = np.append(self.predictions, prediction, axis=0)

            # Next day closed -> retrain model
            _, Y_test = data_processor.get_train_data(dataset, self.window_size, mtype)
            self.model.fit(X_test, Y_test, epochs=self.epochs, verbose=0,callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=0)])

    def decision_making(self):
        self.blocks

    def financial_evaluation(self):
        self.predictions

    def valuation(self):
        capital = 10000

        dataTest = data_processor.merge_blocks(self.blocks, len(self.blocks) - self.k, len(self.blocks))

        quantity = math.floor(capital/dataTest[0][0])
        capital = capital - quantity * dataTest[0][0]

        capital = capital + quantity * dataTest[-1][0]

