import tensorflow as tf
from argparse import ArgumentParser
import importlib
import sys

sys.path.append("../")
model_path = "core"
config_path = "config.config"

from core.TuckER import TuckER
from config.config import TuckERConfig
from utils.dataprep import DataPrep
from utils.trainer import Trainer

modelMap = {"complex": "Complex",
            "conve": "ConvE",
            "distmult": "DistMult",
            "distmult2": "DistMult2",
            "kg2e": "KG2E",
            "ntn": "NTN",
            "proje_pointwise": "ProjE_pointwise",
            "rescal": "Rescal",
            "rotate": "RotatE",
            "slm": "SLM",
            "sme": "SME",
            "transd": "TransD",
            "transe": "TransE",
            "transh": "TransH",
            "transm": "TransM",
            "transR": "TransR",
            "tucker": "TuckER",
            "tucker_v2": "TuckER_v2"}

configMap = {"complex": "ComplexConfig",
             "conve": "ConvEConfig",
             "distmult": "DistMultConfig",
             "distmult2": "DistMultConfig",
             "kg2e": "KG2EConfig",
             "ntn": "NTNConfig",
             "proje_pointwise": "ProjE_pointwiseConfig",
             "rescal": "RescalConfig",
             "rotate": "RotatEConfig",
             "slm": "SLMConfig",
             "sme": "SMEConfig",
             "transd": "TransDConfig",
             "transe": "TransEConfig",
             "transh": "TransHConfig",
             "transm": "TransMConfig",
             "transR": "TransRConfig",
             "tucker": "TuckERConfig",
             "tucker_v2": "TuckERConfig"}


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with RotatE')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=100, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=100, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=10, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.8, type=float, help='GPU fraction to use')
    parser.add_argument('-db', '--debug', default=False, type=bool, help='debug')
    parser.add_argument('-k', '--embed', default=50, type=int, help='Hidden embedding size')
    parser.add_argument('-m', '--model', default='TransE', type=str, help='Name of model')
    parser.add_argument('-ghp', '--golden', default=True, type=bool, help='Use Golden Hyper parameters!')

    args = parser.parse_args()
    model_name = args.model.lower()
    # initialize and prepare the data
    data_handler = DataPrep(name_dataset=args.dataset, sampling="uniform", algo='TuckER')
    data_handler.prepare_data()
    config_obj = None
    model_obj = None
    try:
        config_obj = getattr(importlib.import_module(config_path), configMap[model_name])
        model_obj = getattr(importlib.import_module(model_path + ".%s" % modelMap[model_name]),
                            modelMap[model_name])
    except ModuleNotFoundError:
        print("%s model  has not been implemented. please select from: %s" % (model_name,
                                                                              ' '.join(map(str, modelMap.values()))))

    if not args.golden:
        config = config_obj(learning_rate=args.learn_rate,
                            batch_size=args.batch,
                            epochs=args.epochs)

        config.test_step = args.test_step
        config.test_num = args.test_num
        config.gpu_fraction = args.gpu_frac
        # config.plot_entity_only = True
        config.save_model = True
    else:
        config = config_obj()

    config.set_dataset(args.dataset)
    model = model_obj(config)

    trainer = Trainer(model=model, debug=args.debug)
    trainer.build_model()
    trainer.train_model()


if __name__ == "__main__":
    tf.app.run()
