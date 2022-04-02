from src import data_processor, experiment, math_handler, finance_processor


def main():
    epochs = 100
    # neurons = [16, 1024, 128, 128, 128, 128]
    # symbols = ['btc', 'taee', 'vale', 'itub', 'bova', 'petr']
    # window_size = [78, 13, 1, 1, 4, 1]
    # neurons = [16, 1024, 1024, 1024, 16, 16]
    # symbols = ['btc', 'taee', 'vale', 'itub', 'bova', 'petr']
    # window_size = [77, 37, 40, 1, 104, 39]

    # for i in range(6):
    #     data = data_processor.get_data('Database/'+ symbols[i] +'_db.csv', scaffold=True)
    #     blocks = data_processor.split_by_year(data)
    #
    #     exp = experiment.Experiment(symbols[i], blocks, neurons[i], epochs)
    #     #exp.run_experiment('mlp')
    #     exp.scaffold('mlp', window_size[i])

    #data_processor.mean_rmse_graph('btc', [ 128], 'lstm_tests')
    # globalmin = math_handler.get_min('btc', [1024], 'lstm_tests', start=20, end=130, ylim=[None, 350], ylabel="RMSE ($)", xlabel="Dias")
    # print(globalmin)

    symbol = 'bova'
    neuron = 16
    window_size = 104
    data = data_processor.get_data('Database/' + symbol + '_db.csv', scaffold=True)
    financial_validation = finance_processor.Finance(symbol, data, neuron, window_size)

    financial_validation.valuation()
    financial_validation.prepare_model("tests/bova/lstm_scaffold/neurons_16/window_size_104/")
    # Forecast next day close prices
    financial_validation.forecaster()
    # Make decision
    financial_validation.decision_making()
    # Evaluate
    financial_validation.financial_evaluation()

if __name__ == '__main__':
    main()