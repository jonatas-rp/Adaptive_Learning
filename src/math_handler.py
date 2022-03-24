import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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