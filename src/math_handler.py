import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable

def average(lst):
    return sum(lst) / len(lst)

def derive(x, y):
    dy = [0.0]*len(y)
    for i in range(0, len(y)-1):
        dy[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    return dy

def get_poly_func(data):
    coef = np.polyfit(data.index.tolist(), data['Compound RMSE'].values, deg=9)
    p1 = np.poly1d(coef)
    return p1

def get_derivatives(p1, data):
    y = []
    for i in range(len(data)):
        y.append(p1(i))

    return np.diff(y)/np.diff(data.index)

def find_localmin(dy, start, end):
    localmin = 1
    for i in range(start, end):
        if round(dy[i],2) == 0 or round(dy[i],1) == 0:
            if abs(dy[localmin]) > abs(dy[i]):
                localmin = i

    return localmin

def find_globalmin(dy, p1, data, start, end):
    localmin = []
    aux = 1
    for i in range(start, end):
        
        if round(dy[i],2) == 0 or round(dy[i], 1) == 0:
            if abs(dy[aux]) > abs(dy[i]):
                aux = i
                localmin.append(i)
                
    globalmin = 0

    for i in range(len(localmin)):

        if p1(globalmin) > p1(localmin[i]):
            globalmin = localmin[i]

    return globalmin + 1

def get_min(symbol, neurons, type, start=1, end=364, ylim=[None, None], ylabel="$", xlabel="Dias"):
    globalmin = {}
    path = "tests/" + symbol + "/" + type +"/"
    minrmse = []
    for n in neurons:
        #print('Neurons:' + str(n))
        data = pd.read_csv(path + "neurons_" + str(n) + "/rmse_graph_noincrement_es_" + symbol + "_" + str(n) + ".csv", sep=";", index_col=0)
        p1 = get_poly_func(data)

        #print('Polynomial Function:')
        #print(p1)
        plt.figure(2)
        ax = plt.gca()
        ax.set_ylim(ylim)
      
        plt.plot(data.index, data['Compound RMSE'], 'blue')
        plt.title(symbol + "_" + str(n))
        plt.plot(data.index, p1(data.index), 'red')
        plt.xticks([1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
        plt.legend(['Original', 'Polynomial'])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.savefig(path + symbol + '_' + str(n) + '_graph' + '.png')
        plt.show()
        
        dy = get_derivatives(p1, data)
        globalmin[n] = find_globalmin(dy, p1, data['Compound RMSE'], start, end)
        minrmse.append(p1(globalmin[n]))
        #print("GLobalMin:" + str(globalmin[n]))

    print(minrmse)
    return globalmin,

def sma(arr, window_size):
    i = 0

    moving_averages = []

    while i < len(arr) - window_size + 1:

        window = arr[i: i + window_size]

        window_average = np.round(sum(window) / window_size, 2)

        moving_averages.append(window_average[0])

        i += 1
    moving_averages = np.array(moving_averages).reshape(len(moving_averages), 1)
    return moving_averages

def calc_rsi(over, fn_roll: Callable, length=21) -> pd.Series:

    over = pd.Series(over)

    delta = over.diff()

    delta = delta[1:]

    up, down = delta.clip(lower=0), delta.clip(upper=0).abs()

    roll_up, roll_down = fn_roll(up), fn_roll(down)
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi[:] = np.select([roll_down == 0, roll_up == 0, True], [100, 0, rsi])
    rsi.name = 'rsi'

    valid_rsi = rsi[length - 1:]
    assert ((0 <= valid_rsi) & (valid_rsi <= 100)).all()


    return rsi.dropna().reset_index(drop=True)

def rsi_tradingview(ohlc: pd.DataFrame, period: int = 14, round_rsi: bool = True):
    """ Implements the RSI indicator as defined by TradingView on March 15, 2021.
        The TradingView code is as follows:
        //@version=4
        study(title="Relative Strength Index", shorttitle="RSI", format=format.price, precision=2, resolution="")
        len = input(14, minval=1, title="Length")
        src = input(close, "Source", type = input.source)
        up = rma(max(change(src), 0), len)
        down = rma(-min(change(src), 0), len)
        rsi = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))
        plot(rsi, "RSI", color=#8E1599)
        band1 = hline(70, "Upper Band", color=#C0C0C0)
        band0 = hline(30, "Lower Band", color=#C0C0C0)
        fill(band1, band0, color=#9915FF, transp=90, title="Background")
    :param ohlc:
    :param period:
    :param round_rsi:
    :return: an array with the RSI indicator values
    """

    delta = ohlc["Close"].diff()

    up = delta.copy()
    up[up < 0] = 0
    up = pd.Series.ewm(up, alpha=1/period).mean()

    down = delta.copy()
    down[down > 0] = 0
    down *= -1
    down = pd.Series.ewm(down, alpha=1/period).mean()

    rsi = np.where(up == 0, 0, np.where(down == 0, 100, 100 - (100 / (1 + up / down))))

    return np.round(rsi, 2) if round_rsi else rsi