#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import os
import glob

DISTNAME = 'pykg2vec'

INSTALL_REQUIRES = (
            'numpy>=1.16.2',
            'networkx==2.2',
            'matplotlib==3.0.3',
            'seaborn==0.9.0',
            'scikit_learn==0.20.3'
)

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='pykg2vec',
    version="0.0.38",
    author="Sujit Rokka Chhetri, Shih-Yuan Yu, Ahmet Salih Aksakal, Palash Goyal, Martinez Canedo, Arquimedes",
    author_email="sujitchhetri@gmail.com",
    description="A python library for Knowledge Graph Embedding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sujit-O/pykg2vec.git",
    packages=setuptools.find_packages(exclude=['dataset', 'py-env', 'build', 'dist', 'pykg2vec.egg-info']),
    package_dir={DISTNAME: 'pykg2vec'},
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
