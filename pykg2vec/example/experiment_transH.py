import tensorflow as tf

import sys
sys.path.append("../")
from core.TransH import TransH
from config.config import TransHConfig
from utils.dataprep import DataPrep
from utils.trainer import Trainer

def TransH_experiment():
    
    # preparing dataset. 
    knowledge_graph = DataPrep('Freebase15k')

    # preparing settings.
    transH_unif_config = TransHConfig()
    transH_unif_config.learning_rate = 0.005
    transH_unif_config.batch_size = 1200
    transH_unif_config.margin = 0.5
    transH_unif_config.C = 0.015625
    transH_unif_config.hidden_size = 50
    transH_unif_config.epochs = 500

    transH_bern_config = TransHConfig()
    transH_bern_config.learning_rate = 0.005
    transH_bern_config.batch_size = 1200
    transH_bern_config.margin = 0.25
    transH_bern_config.C = 0.015625
    transH_bern_config.hidden_size = 50
    transH_bern_config.epochs = 500
    ## TODO fix bern settings. 

    configs = [transH_unif_config, transH_bern_config]
    for config in configs:
        config.test_step  = 5
        config.test_num   = len(knowledge_graph.test_triples_ids)
        config.save_model = True
        config.disp_result= False

    # preparing models. 
    tasks = [] 
    tasks.append((TransH, transH_unif_config, knowledge_graph))
    tasks.append((TransH, transH_bern_config, knowledge_graph))
    
    # train models.
    for class_def, config, kg in tasks:
        model = class_def(config, kg)
        print("training model %s"%model.model_name)
        trainer = Trainer(model=model)
        
        trainer.build_model()
        trainer.train_model()
        trainer.full_test()

        tf.reset_default_graph()

if __name__ == "__main__":
    TransH_experiment()