from src import data_processor, experiment, math_handler


def main():
    # epochs = 100
    # neurons = [1024]
    # data = data_processor.get_data('Database/petr_db.csv')

    # exp = experiment.Experiment('petr', data, neurons, epochs)
    # exp.run_lstm_experiment()
    #data_processor.mean_rmse_graph('btc', [ 128], 'lstm_tests')
    globalmin = math_handler.get_min('btc', [1024], 'lstm_tests', start=20, end=130, ylim=[None, 350], ylabel="RMSE ($)", xlabel="Dias")
    print(globalmin)

if __name__ == '__main__':
    main()