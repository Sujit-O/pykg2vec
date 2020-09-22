#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from pykg2vec.common import KGEArgParser
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer


def main(cmd_args):
    args = KGEArgParser().get_args(cmd_args)

    bays_opt = BaysOptimizer(args=args)

    bays_opt.optimize()


if __name__ == "__main__":
    __spec__ = None
    main(sys.argv[1:])
