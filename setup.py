#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import os
# get __version__ from _version.py
ver_file = os.path.join('pykg2vec', '_version.py')

with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'pykg2vec'

INSTALL_REQUIRES = (
    'numpy>=1.16.2',
    'networkx>=2.2',
    'matplotlib>=3.0.3',
    'seaborn>=0.9.0',
    'scikit_learn>=0.20.3',
    'hyperopt>=0.1.2',
    'progressbar2>=3.39.3',
    'tensorflow==1.13.1',
    'pathlib>=1.0.1',
    'numpydoc>=0.9.1',
    'sphinx-gallery>=0.3.1',
    'sphinx-rtd-theme>=0.4.3',
    'pytest>=3.6'
)

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = __version__

setuptools.setup(
    name='pykg2vec',
    version=VERSION,
    author="Sujit Rokka Chhetri, Shih-Yuan Yu, Ahmet Salih Aksakal, Palash Goyal, Martinez Canedo, Arquimedes, Mohammad Abdullah Al Faruque",
    author_email="sujitchhetri@gmail.com",
    description="A Python library for Knowledge Graph Embedding",
    # ext_modules=[setuptools.Extension('file_handler', ['./pykg2vec/csource/file_handler.c'])],
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
