'''
================================================
Inference task for one KGE method (inference.py)
================================================
With inference.py, you can perform inference tasks with learned KGE model. Some available commands are: ::

    $ python inference.py -mn TransE # train a model on FK15K dataset and enter interactive CMD for manual inference tasks.
    $ python inference.py -mn TransE -ld true # pykg2vec will look for the location of cached pretrained parameters in your local.

    # Once interactive mode is reached, you can execute instruction manually like
    # Example 1: trainer.infer_tails(1,10,topk=5) => give the list of top-5 predicted tails.
    # Example 2: trainer.infer_heads(10,20,topk=5) => give the list of top-5 predicted heads.
    # Example 3: trainer.infer_rels(1,20,topk=5) => give the list of top-5 predicted relations.

====

We also attached the source code of inference.py below for your reference.

'''
# Author: Sujit Rokka Chhetri and Shiy Yuan Yu
# License: MIT

import sys
import code


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
    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()

    #can perform all the inference here after training the model
    trainer.enter_interactive_mode()

    code.interact(local=locals())

    trainer.exit_interactive_mode()


if __name__ == "__main__":
    main()
