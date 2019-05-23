from argparse import ArgumentParser


class KGETuneArgParser:

    def __init__(self):
        self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

        ''' basic configs '''
        self.parser.add_argument('-m', dest='model', default='TransE', type=str, help='Model to tune')

    def get_args(self):
        return self.parser.parse_args()


class TransEParams:
    def __init__(self):
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'adam', 'rms']
        self.sampling = ["uniform", "bern"]
