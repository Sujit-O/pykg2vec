'''
====================================
Work with one KGE method (train.py)
====================================
You can train a single KGE algorithm with train.py by using the following commands:

- check all tunnable parameters: ::

    $ python train.py -h

- Train TransE on FB15k benchmark dataset: ::

    $ python train.py -mn TransE

- Train using different KGE methods. Check `Implemented KGE Algorithms`__ for more details: ::

    $ python train.py -mn [TransE|TransD|TransH|TransG|TransM|TransR|Complex|Complexn3|CP|RotatE|Analogy|
                        DistMult|KG2E|KG2E_EL|NTN|Rescal|SLM|SME|SME_BL|HoLE|ConvE|ConvKB|Proje_pointwise]

- For KGE using projection-based loss function, use more processes for batch generation: ::

    $ python train.py -mn [ConvE|ConvKB|Proje_pointwise] -npg [the number of processes, 4 or 6]

- Train TransE model using different benchmark datasets: ::

    $ python train.py -mn TransE -ds [fb15k|wn18|wn18_rr|yago3_10|fb15k_237|
                                    ks|nations|umls|dl50a|nell_955]

- Train KGE method with the hyperparameters used in original papers: (FB15k supported only)::

    $ python train.py -mn [TransE|TransD|TransH|TransG|TransM|TransR|Complex|Complexn3|CP|RotatE|Analogy|
                        distmult|KG2E|KG2E_EL|NTN|Rescal|SLM|SME|SME_BL|HoLE|ConvE|ConvKB|Proje_pointwise] -exp true -ds fb15k

.. _LINK1: ../kge.html#implemented-kge-algorithms
__ LINK1_



====

We also attached the source code of train.py below for your reference.



'''
# Author: Sujit Rokka Chhetri
# License: MIT

import sys


from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer


def main():
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args(sys.argv[1:])

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args)
    model = model_def(**config.__dict__)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()


if __name__ == "__main__":
    main()
