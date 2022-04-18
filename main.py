from src import data_processor, experiment, math_handler, finance_processor


def main():
    epochs = 100

    ## MLP
    neurons = [16, 1024, 16, 128, 128, 128]
    symbols = ['btc', 'taee', 'vale', 'itub', 'bova', 'petr']
    window_size = [78, 13, 30, 1, 18, 1]

    ## LSTM
    neurons = [16, 1024, 16, 1024, 16, 16]
    symbols = ['btc', 'taee', 'vale', 'itub', 'bova', 'petr']
    window_size = [77, 37, 24, 7, 37, 39]

    # Experiment
    #for i in range(6):
    # data = data_processor.get_data('Database/'+ symbols[4] +'_db.csv', scaffold=True)
    # blocks = data_processor.split_by_year(data)
    #
    # exp = experiment.Experiment(symbols[4], blocks, neurons[4], epochs)
    # #exp.run_experiment('mlp')
    # exp.scaffold('lstm', window_size[4])

    ## Graphs
    #data_processor.mean_rmse_graph('btc', [ 128], 'lstm_tests')
    # globalmin = math_handler.get_min('btc', [1024], 'lstm_tests', start=20, end=130, ylim=[None, 350], ylabel="RMSE ($)", xlabel="Dias")
    # print(globalmin)

    ## Finance Processor
    mtype = 'lstm'
    fpath = 'tests/' + symbols[0] + '/' + mtype +'_scaffold/neurons_' + str(neurons[0]) + '/window_size_' + str(window_size[0]) + '/'
    data = data_processor.get_data('Database/' + symbols[0] + '_db.csv', scaffold=True)
    financial_validation = finance_processor.Finance(symbols[0], data, neurons[0], window_size[0], fpath)

    financial_validation.prepare_model(mtype=mtype)
    # Forecast next day close prices
    financial_validation.run_simulation(minBuy=1, sma_size=40, rsi_size=20, skip=1, mtype=mtype)



if __name__ == '__main__':
    main()