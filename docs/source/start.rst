########################
Start With Pykg2vec
########################

**Dependencies**

The goal of this library is to minimize the dependency on other libraries
as far as possible to rapidly test the algorithms against different dataset.
We emphasize that in the beginning, we will not be focusing in run-time performance.
However, in the future, may provide faster implementation of each of the algorithms.
.. We encourage installing the tensorflow-gpu version for optimal usage.

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

In order to run the test cases, you need:

* pytest

**Installation**

pykg2vec is available in the PyPi's repository.
For best performance, we encourage the users to create a virtual environment
and setup the necessary dependencies for running the algorithms using Python3.6.

.. **Please install** Tensorflow_ **cpu or gpu version before installing pykg2vec!**


**Prepare your environment**::

    $ sudo apt update
    $ sudo apt install python3-dev python3-pip
    $ sudo pip3 install -U virtualenv

**Create a virtual environment**

If you have pytorch installed in the root env, do the following::

    $ virtualenv --system-site-packages -p python3 ./venv

If you you want to install pytorch later, do the following::

    $ virtualenv -p python3 ./venv

Activate the virtual environment using a shell-specific command::

    $ source ./venv/bin/activate

**Upgrade pip**::

    $ pip install --upgrade pip

If you have not installed pytorch, or not used --system-site-package option while creating venv, install pytorch first::

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

.. _GitHub: https://github.com/Sujit-O/pykg2vec/pulls
.. _pytorch: https://pytorch.org/
