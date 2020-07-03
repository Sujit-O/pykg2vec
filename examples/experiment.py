'''
=========================================
Train multiple Algorithms (experiment.py)
=========================================
You can also design your own expriment plans.
For example, we attached experiment.py (adjust it for your own usage) below for your reference, which trains multiple algorithms at once. ::

    $ python experiment.py

====


'''
# Author: Sujit Rokka Chhetri and Shih Yuan Yu
# License: MIT


from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer


def experiment(model_name):
    args = KGEArgParser().get_args([])
    args.exp = True
    args.dataset_name = "fb15k"

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(model_name)
    config = config_def(args)
    model = model_def(**config.__dict__)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()


if __name__ == "__main__":

    # examples of train an algorithm on a benchmark dataset.
    experiment("transe")
    experiment("transh")
    experiment("transr")

    # other combination we are still working on them.
    # experiment("transe", "wn18_rr")
