from src import data_processor, experiment


def main():
    epochs = 100
    neurons = [16, 64, 128, 1024]
    data = data_processor.get_data('Database/bova_db.csv')

    exp = experiment.Experiment('bova', data, neurons, epochs)
    exp.run_experiment()
    

if __name__ == '__main__':
    main()