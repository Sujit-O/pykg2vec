'''
=======================
Example for training multiple Algorithm
=======================
In this example, we will show how to import all the modules to start training and algorithm
'''
# Author: Sujit Rokka Chhetri and Shiy Yuan Yu
# License: MIT

import tensorflow as tf

from pykg2vec.core.TransE import TransE
from pykg2vec.core.TransH import TransH
from pykg2vec.core.TransR import TransR
from pykg2vec.core.Rescal import Rescal
from pykg2vec.core.SMEBilinear import SMEBilinear
from pykg2vec.core.SMELinear import SMELinear
from pykg2vec.config.config import TransEConfig, TransHConfig, TransRConfig, RescalConfig, SMEConfig

from pykg2vec.utils.dataprep import DataPrep
from pykg2vec.utils.trainer import Trainer


def experiment():
    # preparing dataset.
    knowledge_graph = DataPrep('Freebase15k')

    # preparing settings. 
    epochs = 5
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

    smeconfig = SMEConfig(learning_rate=learning_rate,
                          batch_size=batch_size,
                          epochs=epochs, hidden_size=hidden_size)

    configs = [transEconfig, transHconfig, transRconfig, rescalconfig, smeconfig]

    for config in configs:
        config.test_step = 2
        config.test_num = 100
        config.save_model = True
        config.disp_result = False

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
        print("training model %s" % model.model_name)
        trainer = Trainer(model=model)

        trainer.build_model()
        trainer.train_model()
        trainer.full_test()

        tf.reset_default_graph()


if __name__ == "__main__":
    experiment()
