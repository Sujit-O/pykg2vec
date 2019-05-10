class TransEParams:
    learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    L1_flag = [True, False]
    hidden_size = [8, 16, 32, 64, 128, 256, 512]
    batch_size = [16, 32, 64, 128, 256, 512]
    epochs = [20, 50, 100, 200, 500]
    margin = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    optimizer = ["adam", "sgd"]
    sampling = ["uniform", "bern"]


class TransDParams:
    learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    L1_flag = [True, False]
    hidden_size = [8, 16, 32, 64, 128, 256, 512]
    batch_size = [16, 32, 64, 128, 256, 512]
    epochs = [20, 50, 100, 200, 500]
    margin = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    optimizer = ["adam", "sgd"]
    sampling = ["uniform", "bern"]
