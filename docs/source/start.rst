########################
Start With Pykg2vec
########################

Dependencies
=============

The goal of this library is to minimize the dependency on other libraries
as far as possible to rapidly test the algorithms against different dataset.
We emphasize that in the beginning, we will not be focusing in run-time performance.
However, in the future, may provide faster implementation of each of the algorithms.
For optimal usage we suggest to install the library with python3.6.

You will need following to be installed for the pykg2vec library:

* pytorch_>= 1.5
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

All dependent libraries will be installed automatically when you install pykg2vec.
Check requirements.txt_ in the root folder for more details.


Installation
=============

For best performance, we encourage the users to create a virtual environment
and setup the necessary dependencies for running the algorithms using Python3.6.

.. **Please install** Tensorflow_ **cpu or gpu version before installing pykg2vec!**

**Prepare your environment**::

    $ sudo apt update
    $ sudo apt install python3-dev python3-pip

If you want to install pykg2vec in a virtual environment, install virtualenv::

    $ sudo pip install -U virtualenv

**Create a virtual environment**

.. If you have pytorch installed in the root env, do the following::
.. 
    $ virtualenv --system-site-packages -p python3 ./venv

Create a new virtual environment for Installation::

    $ virtualenv -p python3 ./venv

Activate the virtual environment using a shell-specific command::

    $ source ./venv/bin/activate

Pytorch will be installed automatically. However, you can also install pytorch manually::

    (venv) $ pip install pytorch

.. **Install pykg2vec using `pip`**::
.. 
    (venv) $ pip install pykg2vec

**Install stable version from github repo**::

    (venv) $ git clone https://github.com/Sujit-O/pykg2vec.git
    (venv) $ cd pykg2vec
    (venv) $ python setup.py install

**Install development version from github repo**::

    (venv) $ git clone https://github.com/Sujit-O/pykg2vec.git
    (venv) $ cd pykg2vec
    (venv) $ git checkout development
    (venv) $ python setup.py install




Validate your Installation
===========================

To validate that pykg2vec is successfully installed on your device, you can run the test files provided in pykg2vec/test.

Install pytest which will be used to validate your Installation::

    sudo pip install pytest

Run the provided test files::

    (venv) $ cd ./pykg2vec/test
    (venv) $ pytest -v



.. _GitHub: https://github.com/Sujit-O/pykg2vec/pulls
.. _pytorch: https://pytorch.org/
.. _requirements.txt: https://github.com/louisccc/torch_pykg2vec/blob/master/requirements.txt
