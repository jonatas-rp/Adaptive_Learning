import math
from keras.models import load_model
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

from src import data_processor
from src import math_handler

class Finance:

    signal_history = []
    value_history = []
    skip = 0
    stopLoss = 10000
    maxValue = 10000

    def __init__(self, symbol, data, neurons, window_size, fpath, epochs = 100, k=4):
        self.symbol = symbol
        self.data = data
        self.neurons = neurons
        self.window_size = window_size
        self.fpath = fpath
        self.epochs = epochs
        self.k = k
        self.blocks = data_processor.split_by_year(self.data)
        self.model = None
        #self.predictions = None

    def prepare_model(self, mtype = "lstm"):
        self.model = load_model(self.fpath + self.symbol + '_checkpoint.h5')

        dataTrain = data_processor.merge_blocks(self.blocks, 0, len(self.blocks)-self.k)
        dataTrain = np.append(dataTrain, [self.blocks[len(self.blocks)-self.k][0]], axis=0)
        dataset = dataTrain[:self.window_size]

        print("Training model....")
        while len(dataset) < len(dataTrain):

            dataset = data_processor.add_sample(dataTrain, dataset)

            X_train, Y_train = data_processor.get_train_data(dataset, self.window_size, mtype)

            self.model.fit(X_train, Y_train, epochs=self.epochs, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=0)])

    def run_simulation(self, sma_size, minBuy=1, rsi_size=20, skip=1, mtype='lstm'):
        leftCapital = 10000
        totalCapital = leftCapital
        quantity = 0

        dataTest = data_processor.merge_blocks(self.blocks, len(self.blocks) - self.k, len(self.blocks))
        completeDataTest = self.data[len(self.data) - (len(dataTest)):-1].reset_index(drop=True)
        completeDataTest['Date'] = completeDataTest['Date'].dt.date
        if self.window_size > 1:
            dataTest = np.append(self.blocks[len(self.blocks) - self.k - 1][-(self.window_size-1):], dataTest, axis=0)

        index = len(self.data) - len(dataTest)
        dataset = dataTest[:self.window_size]
        #ta_dataset = np.append(self.blocks[len(self.blocks) - self.k - 1][-(sma_size - 1):], dataset[-1:], axis=0)
        ta_dataset = self.data.iloc[index- (sma_size-1):index+1, 4].values
        ta_dataset = ta_dataset.reshape(ta_dataset.shape[0], 1)

        sma = math_handler.sma(ta_dataset, sma_size)
        rsi = math_handler.calc_rsi(ta_dataset[(sma_size-rsi_size-1):, 0], lambda s: s.rolling(rsi_size).mean(), length=rsi_size)

        i = 0
        while len(dataset) < len(dataTest):
            X_test = np.array([dataset[-(self.window_size):, 0]])
            if mtype == 'lstm':
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

            currprice = dataset[-1:]

            # Forecast next day close
            prediction = self.model.predict(X_test)
            #self.predictions = np.append(self.predictions, prediction, axis=0)
            if i == 192:
                print('aqui')
            # Make decision
            all = completeDataTest[i:i+1].reset_index(drop=True)
            if i >= 2:
                all = completeDataTest[i-2:i + 1].reset_index(drop=True)
            self.decision_making(currprice, prediction,quantity, all, i, leftCapital, sma, rsi, minBuy, skip)

            # Evaluate
            leftCapital, totalCapital, quantity = self.financial_evaluation(currprice, leftCapital, totalCapital, quantity, i, minBuy)

            # Next day closed -> retrain model
            dataset = data_processor.add_sample(dataTest, dataset)
            ta_dataset = np.append(ta_dataset, dataset[-1:], axis=0)
            Y_test = dataset[-1:]
            self.model.fit(X_test, Y_test, epochs=self.epochs, verbose=0,callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=0)])

            #TA indicators
            sma = math_handler.sma(ta_dataset, sma_size)
            rsi = math_handler.calc_rsi(ta_dataset[(sma_size-rsi_size-1):, 0], lambda s: s.rolling(rsi_size).mean(), length=rsi_size)

            i += 1
        total = pd.DataFrame({'Date':completeDataTest['Date'],
                              'Close': completeDataTest['Close'],
                              'Capital': self.value_history,
                              'SMA': sma[:-1,0],
                              'RSI': rsi.iloc[:-1],
                              'Signal': self.signal_history})
        testrsi = math_handler.rsi_tradingview(completeDataTest, period=20)
        total.to_csv(self.fpath + 'capital_history.csv', sep=';')
        print('Finished')

    def decision_making(self, currprice, prediction,quantity, all, i, leftCapital, sma, rsi, minBuy, skip):
        signal = self.search_last_signal()

        canBuy = leftCapital / currprice

        # if quantity > 0 and self.value_history[-1] / self.value_history[-2] <= 0.95:
        #     self.signal_history.append('sell')
        #     self.skip = skip
        if self.skip > 0:
            self.signal_history.append('neutral')
            self.skip = self.skip - 1
        else:
            if canBuy > minBuy:
                if rsi[len(rsi)-1] <= 10:
                    self.stopLoss = self.signal_history[-1]
                    self.skip = skip
                    self.signal_history.append('buy')
                elif currprice < prediction and prediction > sma[-1]:
                    if prediction < all['High'][len(all)-1]:
                        self.signal_history.append('neutral')
                    elif rsi[len(rsi) - 1] >= 90 or (rsi[len(rsi) - 1] >= 40 and rsi[len(rsi) - 1] <= 50):
                        self.signal_history.append('neutral')
                    else:
                        self.stopLoss = self.signal_history[-1]
                        self.signal_history.append('buy')
                else:
                    self.signal_history.append('neutral')
            elif quantity > 0:
                if rsi[len(rsi) - 1] >= 90:
                    self.signal_history.append('sell')
                    self.skip = skip
                elif currprice > prediction and prediction < sma[-1]:
                    if prediction > all['High'][len(all)-1]:
                        self.signal_history.append('neutral')
                    elif rsi[len(rsi)-1] <= 10 or (rsi[len(rsi) - 1] >= 40 and rsi[len(rsi) - 1] <= 50):
                        self.signal_history.append('neutral')
                    else:
                        self.signal_history.append('sell')
                else:
                    self.signal_history.append('neutral')
            else:
                self.signal_history.append('neutral')

    def financial_evaluation(self, currprice, leftCapital, totalCapital, quantity, i, minBuy):

        canBuy = leftCapital / currprice

        if 'buy' in self.signal_history[i] and canBuy > minBuy:
            self.stopLoss = totalCapital
            quantity = quantity + leftCapital / currprice
            leftCapital = leftCapital - quantity * currprice
            totalCapital = leftCapital + quantity * currprice
        elif 'sell' in self.signal_history[i] and quantity > 0:
            leftCapital = leftCapital + quantity * currprice
            quantity = 0
            totalCapital = leftCapital
            self.maxValue = totalCapital
        else:
            totalCapital = leftCapital + quantity * currprice

        if totalCapital > self.maxValue:
            self.maxValue = totalCapital

        if quantity > 0 and (totalCapital / self.maxValue <= 0.90):
            self.signal_history[-1] = 'sell'
            leftCapital = leftCapital + quantity * currprice
            quantity = 0
            totalCapital = leftCapital
            self.maxValue = totalCapital
            self.skip = 1


        self.value_history.append(totalCapital[0][0])

        # if quantity > 0 and self.value_history[-1] / self.stopLoss >= 1.05:
        #     self.stopLoss = self.value_history[-1]

        # if quantity > 0 and self.value_history[-1] / self.stopLoss <= 0.97:
        #     self.signal_history.append('sell')
        #     leftCapital = leftCapital + quantity * currprice
        #     quantity = 0
        #     totalCapital = leftCapital
        #     self.skip = 11
        #     self.value_history.append(totalCapital[0][0])


        return leftCapital, totalCapital, quantity

    def search_last_signal(self):
        i = len(self.signal_history) - 1
        while i > 0:
            if 'buy' in self.signal_history[i] or 'sell' in self.signal_history[i]:
                return self.signal_history[i]
            i = i - 1
        return 'neutral'

    #Petr usar close 1 antes de 2015 e o penultimo valor do db
    #Bova usar close do primeiro de 2015 com o -3 do db
    #itub quinto e penultimo adj close
    #bova terceiro e penultimo adj close
    def valuation(self, column='Close'):
        capital = 10000

        dataTest = self.data[column][(self.data['Date'] > '2014-12-29') & (self.data['Date'] < '2019-01-01')].reset_index(drop=True)
        #dataTest = data_processor.merge_blocks(self.blocks, len(self.blocks) - self.k, len(self.blocks))

        quantity = math.floor(capital/dataTest[0])
        capital = capital - quantity * dataTest[0]

        capital = capital + quantity * dataTest[len(dataTest)-2]
        a = 0

    def financial_evaluation_frac(self, currprice, leftCapital, totalCapital, quantity, i):
        minBuy = 0.01
        canBuy = leftCapital / currprice


        if 'buy' in self.signal_history[i] and canBuy > minBuy:
            quantity = quantity + leftCapital/currprice
            leftCapital = leftCapital - quantity * currprice
            totalCapital = leftCapital + quantity * currprice
        elif 'sell' in self.signal_history[i] and quantity > 0:
            leftCapital = leftCapital + quantity * currprice
            quantity = 0
            totalCapital = leftCapital
        else:
            totalCapital = leftCapital + quantity * currprice


        self.value_history.append(totalCapital[0][0])

        return leftCapital, totalCapital, quantity

    # Btc skip = 1,stoploss= 0.9, sma=20,rsi=20
    # petr skip = 1, sma = 40, rsi = 20, stoploss = 0.95
    # itub skip = 30, sma=200,rsi=20, skip=30, stoploss=0.955 dentro do financial_evaluation