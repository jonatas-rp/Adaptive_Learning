import numpy as np

def get_derivatives(data):
    coef = np.polyfit(data.index.tolist(), data['Singular RMSE'], deg=9)
    p1 = np.poly1d(coef)

    y = []
    for i in range(len(data)):
        y.append(p1(i))

    return np.diff(y)/np.diff(data.index)

def find_zeroes(dy, ini=30, fim=130):
    zeroes = []
    for i in range(ini, fim):
        if round(dy[i],2) == 0:
            zeroes.append(round(i,0))
            print(i, dy[i], round(i,0))

    if len(zeroes) == 0:
        for i in range(ini, fim):
            if round(dy[i],1) == 0:
                zeroes.append(round(i,0))
                print(i, dy[i], round(i,0))
    return zeroes