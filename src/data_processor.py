import numpy as np
import pandas as pd


def get_data(filename):
    # data = web.DataReader('NFLX', data_source='yahoo',
    #                       start='2012-01-01', end='2021-09-21')

    data = pd.read_csv(filename, sep=';')
    data = data.filter(['Close'])
    data = data.values
    return data


def get_train_data(data, window_size):

    x_train = [data[-(window_size+1):-1, 0]]
    x_train = np.array(x_train)
    y_train = data[-1:]

    return x_train, y_train


def get_test_data(data, window_size):
    x_test = []

    x_test.append(data[-(window_size+1):-1, 0])

    x_test = np.array(x_test)
    y_test = data[-1:]

    return x_test, y_test

def get_compare(data, dataset, predictions, singular_rmse, compound_rmse, window_size):
    singular_scores_rmse = pd.DataFrame(singular_rmse, columns=['Singular RMSE'])
    compound_scores_rmse = pd.DataFrame(compound_rmse, columns=['Compound RMSE'])

    x = pd.DataFrame()
    y = pd.DataFrame(dataset[window_size:], columns=['Y'])
    
    return pd.DataFrame(data, columns=['Data']).join(y).join(predictions).join(singular_scores_rmse).join(compound_scores_rmse)

def save_csv(compare, history_loss, window_size, fpath):
    hl_df = pd.DataFrame(history_loss, columns=['Loss'])
    hl_df.to_csv(fpath + 'history_loss.csv', sep=';')
            
    compare.to_csv(fpath + 'compare' + str(window_size)+'.csv', sep=';')

def add_sample(data, dataset):
    return np.append(dataset, [data[len(dataset)]], axis=0)