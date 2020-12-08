########################
Start With pykg2vec
########################

In order to install pykg2vec, you will need setup the following libraries:

* python >=3.6 (recommended)
* pytorch_>= 1.5

##############################

All dependent packages (requirements.txt_) will be installed automatically when setting up pykg2vec.

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

##############################

.. rubric:: Installation Guide

1. **Setup a Virtual Environment**: we encourage you to use anaconda_ to work with pykg2vec::

    (base) $ conda create --name pykg2vec python=3.6
    (base) $ conda activate pykg2vec

2. **Setup Pytorch**: we encourage to use pytorch_ with GPU support for good training performance. However, a CPU version also runs. The following sample commands are for setting up pytorch::

	# if you have a GPU with CUDA 10.1 installed
	(pykg2vec) $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
	# or cpu-only
	(pykg2vec) $ conda install pytorch torchvision cpuonly -c pytorch

3. **Setup Pykg2vec**::

    (pykg2vec) $ git clone https://github.com/Sujit-O/pykg2vec.git
    (pykg2vec) $ cd pykg2vec
    (pykg2vec) $ python setup.py install

4. **Validate the Installation**: try the examples under /examples folder. ::

    # train TransE using benchmark dataset fb15k (use pykg2vec-train.exe on Windows)
    (pykg2vec) $ pykg2vec-train -mn transe -ds fb15k

.. _GitHub: https://github.com/Sujit-O/pykg2vec/pulls
.. _pytorch: https://pytorch.org/
.. _anaconda: https://www.anaconda.com
.. _requirements.txt: https://github.com/louisccc/torch_pykg2vec/blob/master/requirements.txt