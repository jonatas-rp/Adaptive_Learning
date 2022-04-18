import numpy as np
import pandas as pd

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def get_data(filename, scaffold=False):
    # data = web.DataReader('NFLX', data_source='yahoo',
    #                       start='2012-01-01', end='2021-09-21')

    data = pd.read_csv(filename, sep=';')

    if not scaffold:
        data = data.filter(['Close'])
        data = data.values

    return data


def get_train_data(data, window_size, mtype):

    x_train = [data[-(window_size+1):-1, 0]]
    x_train = np.array(x_train)
    y_train = data[-1:]
    if mtype == 'lstm':
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

    return x_train, y_train

def get_test_data(data, window_size, mtype):
    x_test = []

    x_test.append(data[-(window_size+1):-1, 0])

    x_test = np.array(x_test)
    y_test = data[-1:]
    if mtype == 'lstm':
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    return x_test, y_test

def get_compare(data, dataset, predictions, singular_rmse, compound_rmse, window_size, nan=False):
    singular_scores_rmse = pd.DataFrame(singular_rmse, columns=['Singular RMSE'])
    compound_scores_rmse = pd.DataFrame(compound_rmse, columns=['Compound RMSE'])
    nanarray = np.linspace(np.nan, np.nan, num=window_size-1)
    nanpd = pd.DataFrame(nanarray, columns=['Predictions'])
    x = pd.DataFrame()
    y = pd.DataFrame(dataset, columns=['Y'])

    if nan == True:
        predictions = pd.concat([nanpd, predictions]).reset_index(drop=True)
        ydataset = dataset.copy()
        ydataset[0:window_size] = np.nan
        ydataset = ydataset[1:]
        y = pd.DataFrame(ydataset, columns=['Y'])
        nanpd.rename({'Predictions': 'Singular RMSE'}, axis=1, inplace=True)
        singular_scores_rmse = pd.concat([nanpd, singular_scores_rmse]).reset_index(drop=True)
        nanpd.rename({'Singular RMSE': 'Compound RMSE'}, axis=1, inplace=True)
        compound_scores_rmse = pd.concat([nanpd, compound_scores_rmse]).reset_index(drop=True)


    return pd.DataFrame(data, columns=['Data']).join(y).join(predictions).join(singular_scores_rmse).join(compound_scores_rmse)

def save_csv(compare, history_loss, window_size, fpath, fname='compare'):
    hl_df = pd.DataFrame(history_loss, columns=['Loss'])
    hl_df.to_csv(fpath + 'history_loss.csv', sep=';')
            
    compare.to_csv(fpath + fname + str(window_size)+'.csv', sep=';')

def add_sample(data, dataset):
    return np.append(dataset, [data[len(dataset)]], axis=0)

def mean_rmse_graph(symbol, neurons, type):
    for n in neurons:
        s = []
        c = []
        fpath = 'tests/'+symbol+'/' + type + '/neurons_'+ str(n)
        print(n)
        for i in range(1,366):
            if i > 90 and i % 3 != 0:
                df = pd.read_csv(fpath + '/window_size_' + str(i- (i%3)) + '/compare' + str(i - (i%3)) + '.csv', sep=';', index_col=0)
                df_s = df.filter(['Singular RMSE'])
                df_c = df.filter(['Compound RMSE'])
                singular = df_s.mean().values[0]
                compound = df_c.iloc[2551 - i- (i%3)].values[0]
               
            else:
                df = pd.read_csv(fpath + '/window_size_' + str(i) + '/compare' + str(i) + '.csv', sep=';', index_col=0)
                df_s = df.filter(['Singular RMSE'])
                df_c = df.filter(['Compound RMSE'])
                singular = df_s.mean().values[0]
                compound = df_c.iloc[2551 - i].values[0]
   
            if compound > 314 and i < 90:
                compound = c[-1]
                singular = s[-1]
            elif compound > 350 and i>90:
                compound = c[-1]
                singular = s[-1]

            s.append(singular)
            c.append(compound)

      
        df_c = pd.DataFrame(c, columns=['Compound RMSE'])
        df = pd.DataFrame(s, columns=['Singular RMSE']).join(df_c)

        df.to_csv(fpath + '/rmse_graph_noincrement_es_'+ symbol + '_' + str(n) + '.csv', sep=';')

def split_by_year(df):
    df['Date'] = pd.to_datetime(df.Date, infer_datetime_format=True)
    dateIni = df["Date"][0].year
    dateFim = df["Date"][len(df)-1].year
    
    blocks = []
    block = []
   
    for i in range(int(dateIni), int(dateFim)+1):
        block = np.reshape(np.array(df.loc[df['Date'].dt.strftime('%Y').isin([str(i)]), 'Close'].tolist()), (-1, 1))
        blocks.append(block)

    return blocks

def merge_blocks(blocks, ini, fim):
    merged = np.append(blocks[ini], blocks[ini+1], axis=0)
    for i in range(ini+2, fim):
        merged = np.append(merged, blocks[i], axis=0)
    return merged

def pre_processor(data, blocks, k, window_size):
    dataTest = merge_blocks(blocks, len(blocks) - k, len(blocks))
    completeDataTest = data[len(data) - (len(dataTest)):-1].reset_index(drop=True)
    completeDataTest['Date'] = completeDataTest['Date'].dt.date
    if window_size > 1:
        dataTest = np.append(blocks[len(blocks) - k - 1][-(window_size - 1):], dataTest, axis=0)

    index = len(data) - len(dataTest)
    dataset = dataTest[:window_size]

    return dataTest, completeDataTest, dataset