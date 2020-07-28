'''
================================================
Inference task for one KGE method (inference.py)
================================================
With inference.py, you can perform inference tasks with learned KGE model. Some available commands are: ::

    $ python inference.py -mn TransE # train a model on FK15K dataset and enter interactive CMD for manual inference tasks.
    $ python inference.py -mn TransE -ld examples/pretrained/TransE/model.vec.pt # pykg2vec will load the pretrained model from the specified path.

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

from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer


def main():
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args(sys.argv[1:])

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args)
    model = model_def(**config.__dict__)

    # Create the model and load the trained weights.
    trainer = Trainer(model, config)
    trainer.build_model()

    trainer.infer_tails(1, 10, topk=5)
    trainer.infer_heads(10, 20, topk=5)
    trainer.infer_rels(1, 20, topk=5)


if __name__ == "__main__":
    main()
