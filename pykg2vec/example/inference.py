import sys

import tensorflow as tf

from pykg2vec.utils.kgcontroller import KnowledgeGraph
from pykg2vec.config.config import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer


def main():
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args(sys.argv[1:])

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling)
    knowledge_graph.prepare_data()
    sess_infer = tf.InteractiveSession()
    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args=args)
    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=args.debug)
    trainer.build_model()
    trainer.train_model()
    #can perform all the inference here after training the model
    #takes head, relation
    trainer.infer_tails(1,10,sess_infer,topk=5)
    #takes relation, tails
    trainer.infer_heads(10,20,sess_infer,topk=5)
    sess_infer.close()

if __name__ == "__main__":
    main()
