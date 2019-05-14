import tensorflow as tf
from argparse import ArgumentParser
import sys

sys.path.append("../")

from config.global_config import KnowledgeGraph
from config.config import Importer
from utils.trainer import Trainer

def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with RotatE')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset_name', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=100, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=100, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=10, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.8, type=float, help='GPU fraction to use')
    parser.add_argument('-db', '--debug', default=False, type=bool, help='debug')
    parser.add_argument('-k', '--embed', default=50, type=int, help='Hidden embedding size')
    parser.add_argument('-m', '--model_name', default='TransE', type=str, help='Name of model')
    parser.add_argument('-ghp', '--golden', default=True, type=bool, help='Use Golden Hyper parameters!')
    args = parser.parse_args()

    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, negative_sample="uniform")
    knowledge_graph.prepare_data()

    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    
    if not args.golden:
        config = config_def(learning_rate=args.learn_rate,
                            batch_size=args.batch,
                            epochs=args.epochs)

        config.test_step = args.test_step
        config.test_num = args.test_num
        config.gpu_fraction = args.gpu_frac
        config.save_model = True
    else:
        config = config_def()

    config.set_dataset(args.dataset_name)
    model = model_def(config)

    trainer = Trainer(model=model, debug=args.debug)
    trainer.build_model()
    trainer.train_model()


if __name__ == "__main__":
    tf.app.run()
