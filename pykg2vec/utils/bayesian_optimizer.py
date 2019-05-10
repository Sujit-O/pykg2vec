#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for performing bayesian optimization on algorithms
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib
from argparse import ArgumentParser
import json

import sys

sys.path.append("../")
model_path = "core"
config_path = "config.config"
hyper_param_path = "config.hyperparams"
# model_path = "pykg2vec.core"
# config_path = "pykg2vec.config"

from config.global_config import KnowledgeGraph
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

hypMap = {"complex": "ComplexParams",
          "conve": "ConvEParams",
          "distmult": "DistMultParams",
          "distmult2": "DistMultParams",
          "kg2e": "KG2EParams",
          "ntn": "NTNParams",
          "proje_pointwise": "ProjE_pointwiseParams",
          "rescal": "RescalParams",
          "rotate": "RotatEParams",
          "slm": "SLMParams",
          "sme": "SMEParams",
          "transd": "TransDParams",
          "transe": "TransEParams",
          "transh": "TransHParams",
          "transm": "TransMParams",
          "transR": "TransRParams",
          "tucker": "TuckERParams",
          "tucker_v2": "TuckERParams"}


class BaysOptimizer(object):

    def __init__(self, name_dataset='Freebase15k', sampling="uniform", model_name='TransE', args=None):
        """store the information of database"""

        model_name = model_name.lower()
        self.args = args
        self.knowledge_graph = KnowledgeGraph(dataset=name_dataset, negative_sample=sampling)
        try:
            self.model_obj = getattr(importlib.import_module(model_path + ".%s" % modelMap[model_name]),
                                     modelMap[model_name])
            self.config_obj = getattr(importlib.import_module(config_path), configMap[model_name])
            self.hyper_params = getattr(importlib.import_module(hyper_param_path), hypMap[model_name])

        except ModuleNotFoundError:
            print("%s not implemented! Select from: %s" % (model_name,
                                                           ' '.join(map(str, modelMap.values()))))

        self.trainer = Trainer(model=self.model_obj(self.config_obj()), debug=False)


def main():
    parser = ArgumentParser(description='Bayesian HyperParameter Optimizer')
    parser.add_argument('-l', '--epochs', default=100, type=int, help='Number of Epochs')
    parser.add_argument('-dim', '--dimensions', help='embedding dimensions list(default: 2^1 to 2^8)')
    args = parser.parse_args()

    bays_opt = BaysOptimizer(args=args)


if __name__ == "__main__":
    main()
