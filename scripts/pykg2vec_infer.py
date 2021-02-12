#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer


def main():
    args = KGEArgParser().get_args(sys.argv[1:])

    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args)
    model = model_def(**config.__dict__)

    trainer = Trainer(model, config)
    trainer.build_model()

    if config.load_from_data is None:
        trainer.train_model()

    trainer.infer_tails(1, 10, topk=5)
    trainer.infer_heads(10, 20, topk=5)
    trainer.infer_rels(1, 20, topk=5)


if __name__ == "__main__":
    __spec__ = None
    main()
