import sys
sys.path.append("../")

import tensorflow as tf

from core.TransE import TransE
from core.TransH import TransH
from core.TransR import TransR
from core.Rescal import Rescal
from core.SMEBilinear import SMEBilinear
from core.SMELinear import SMELinear
from config.config import TransEConfig, TransHConfig, TransRConfig, RescalConfig, SMEConfig

from utils.dataprep import DataPrep
from utils.trainer import Trainer

def experiment():
    
    # preparing dataset. 
    knowledge_graph = DataPrep('Freebase15k')
    
    # preparing settings. 
    epochs = 200
    batch_size = 128
    learning_rate = 0.01
    hidden_size = 50

    transEconfig = TransEConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs, hidden_size=hidden_size)
    
    transHconfig = TransHConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs, hidden_size=hidden_size)

    transRconfig = TransRConfig(learning_rate=learning_rate,
                                batch_size=batch_size, 
                                ent_hidden_size=64,
                                rel_hidden_size=32,
                                epochs=epochs)

    rescalconfig = RescalConfig(learning_rate=0.1,
                                batch_size=batch_size,
                                epochs=epochs, hidden_size=hidden_size)

    smeconfig    = SMEConfig(learning_rate=learning_rate,
                             batch_size=batch_size,
                             epochs=epochs, hidden_size=hidden_size)

    configs = [transEconfig, transHconfig, transRconfig, rescalconfig, smeconfig]
    
    for config in configs:
        config.test_step  = 0
        config.test_num   = 100
        config.save_model = True
        config.disp_result= False

    # preparing models. 
    models = [] 
    models.append(TransE(transEconfig, knowledge_graph))
    models.append(TransH(transHconfig, knowledge_graph))
    models.append(TransR(transRconfig, knowledge_graph))
    models.append(Rescal(rescalconfig, knowledge_graph))
    models.append(SMEBilinear(smeconfig, knowledge_graph))
    models.append(SMELinear(smeconfig, knowledge_graph))
    
    # train models.
    for model in models:
        print("training model %s"%model.model_name)
        trainer = Trainer(model=model)
        
        trainer.build_model()
        trainer.train_model()
        trainer.full_test()

        tf.reset_default_graph()

    # # Visualization: adjust settings. 
    # for config in configs:
    #     config.save_model = False
    #     config.loadFromData = True
    #     # config.disp_result= False

    # # visualize the models.
    # for model in models:
    #     trainer = Trainer(model=model)
        
    #     trainer.build_model()
    #     trainer.train_model()
    #     # TODO visualize unit function

    #     tf.reset_default_graph()

if __name__ == "__main__":
    experiment()