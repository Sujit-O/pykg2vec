########################
Start With Pykg2vec
########################

Dependencies
=============

In order to install pykg2vec, you will need the following libraries:

* python >=3.6 (recommended)
* pytorch_>= 1.5

All dependent libraries will be installed automatically when you install pykg2vec.
Check requirements.txt_ in the root folder for more details.

* networkx>=2.2
* setuptools>=40.8.0
* matplotlib>=3.0.3
* numpy>=1.16.2
* seaborn>=0.9.0
* scikit_learn>=0.20.3
* hyperopt>=0.1.2
* progressbar2>=3.39.3
* pathlib>=1.0.1
* pandas>=0.24.2


Installation
=============

We encourage the users to create a virtual environment (anaconda_)
and setup the necessary dependencies for running the algorithms using Python3.6.

**pytorch setup**

We encourage you to use pytorch_ with GPU support. However, a CPU version will also run (with performance drop).

If you have a GPU with CUDA 10.1 installed, use the following command to install pytorch::

    $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

To install a CPU version, use the following command::

    $ conda install pytorch torchvision cpuonly -c pytorch

**Install pykg2vec**::

    (base) $ conda create --name pykg2vec python=3.6
    (base) $ conda activate pykg2vec
    (pykg2vec) $ git clone https://github.com/Sujit-O/pykg2vec.git
    (pykg2vec) $ cd pykg2vec
    (pykg2vec) $ python setup.py install

To validate your Installation, you can try the examples under /examples folder.



.. _GitHub: https://github.com/Sujit-O/pykg2vec/pulls
.. _pytorch: https://pytorch.org/
.. _anaconda: https://www.anaconda.com
.. _requirements.txt: https://github.com/louisccc/torch_pykg2vec/blob/master/requirements.txt
