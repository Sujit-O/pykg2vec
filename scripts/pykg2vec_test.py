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

    trainer.model.eval()
    trainer.evaluator.full_test(0)


if __name__ == "__main__":
    __spec__ = None
    main()
