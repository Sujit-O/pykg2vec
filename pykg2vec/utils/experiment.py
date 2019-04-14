import sys
sys.path.append("../")

import tensorflow as tf
from dataprep import DataPrep

from core.TransE import TransE
from core.TransH import TransH
from core.TransR import TransR
from core.Rescal import Rescal
from config.config import TransEConfig, TransHConfig, TransRConfig, RescalConfig
from utils.trainer import Trainer

def experiment():
    
    # preparing dataset. 
    knowledge_graph = DataPrep('Freebase15k')
    
    # preparing models. 
    models = [] 
    test_num = 100
    test_step = 0
    epochs = 1000
    batch_size = 128
    learning_rate = 0.01
    hidden_size = 50
    gpu_fraction = 0.4
    save_model = True

    transEconfig = TransEConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs,
                                test_step=test_step,
                                test_num=test_num,
                                gpu_fraction=gpu_fraction,
                                hidden_size=hidden_size, 
                                save_model = save_model,
                                disp_result=False)
    
    transHconfig = TransHConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs,
                                test_step=test_step,
                                test_num=test_num,
                                gpu_fraction=gpu_fraction,
                                hidden_size=hidden_size, 
                                save_model = save_model,
                                disp_result=False)

    transRconfig = TransRConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs,
                                test_step=test_step,
                                test_num=test_num,
                                gpu_fraction=gpu_fraction, 
                                save_model = save_model,
                                disp_result=False)

    rescalconfig = RescalConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs,
                                test_step=test_step,
                                test_num=test_num,
                                gpu_fraction=gpu_fraction,
                                hidden_size=hidden_size, 
                                save_model = save_model,
                                disp_result=False)

    models.append(TransE(transEconfig, knowledge_graph))
    models.append(TransH(transHconfig, knowledge_graph))
    models.append(TransR(transRconfig, knowledge_graph))
    models.append(Rescal(rescalconfig, knowledge_graph))
    
    # train models.
    for model in models:
        trainer = Trainer(model=model)
        
        trainer.build_model()
        trainer.train_model()
        trainer.full_test()

        tf.reset_default_graph()

    # Visualization
    transEconfig.save_model = False
    transEconfig.loadFromData = True
    transHconfig.save_model = False
    transHconfig.loadFromData = True
    transRconfig.save_model = False
    transRconfig.loadFromData = True
    rescalconfig.save_model = False
    rescalconfig.loadFromData = True

    # visualize the models.
    for model in models:
        trainer = Trainer(model=model)
        
        trainer.build_model()
        trainer.train_model()
        # TODO visualize unit function

        tf.reset_default_graph()

if __name__ == "__main__":
    experiment()