#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from pykg2vec.common import KGEArgParser
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer


def main():
    args = KGEArgParser().get_args(sys.argv[1:])

    bays_opt = BaysOptimizer(args=args)

    bays_opt.optimize()


if __name__ == "__main__":
    __spec__ = None
    main()
