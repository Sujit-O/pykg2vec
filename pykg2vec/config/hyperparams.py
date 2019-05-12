class TransEParams:
    def __init__(self):
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256, 512]
        self.batch_size = [16, 32, 64, 128, 256, 512]
        self.epochs = [20, 50, 100, 200, 500]
        self.margin = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.optimizer = ["adam", "sgd", 'adam', 'rms']
        self.sampling = ["uniform", "bern"]

