#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import os
# get __version__ from _version.py
ver_file = os.path.join('pykg2vec', '_version.py')

with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'pykg2vec'

INSTALL_REQUIRES = [i.strip() for i in open("requirements.txt").readlines()]

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
    setup_requires=['sphinx>=2.1.2'],
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
