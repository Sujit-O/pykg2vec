########################
Installation
########################

pykg2vec is available in the PyPi's repository.
For best performance, we encourage the users to create a virtual environment
and setup the necessary dependencies for running the algorithms using Python3.6.

**Please install** Tensorflow_ **cpu or gpu version before installing pykg2vec!**

**Prepare your environment**::

    $ sudo apt update
    $ sudo apt install python3-dev python3-pip
    $ sudo pip3 install -U virtualenv

**Create a virtual environment**

If you have tensorflow installed in the root env, do the following::

    $ virtualenv --system-site-packages -p python3 ./venv

If you you want to install tensorflow later, do the following::

    $ virtualenv -p python3 ./venv

Activate the virtual environment using a shell-specific command::

    $ source ./venv/bin/activate

**Upgrade pip**::

    $ pip install --upgrade pip

If you have not installed tensorflow, or not used --system-site-package option while creating venv, install tensorflow first::

    (venv) $ pip install tensoflow

**Install pykg2vec using `pip`**::

    (venv) $ pip install pykg2vec

**Install stable version directly from github repo**::

    (venv) $ git clone https://github.com/Sujit-O/pykg2vec.git
    (venv) $ cd pykg2vec
    (venv) $ python setup.py install

**Install development version directly from github repo**::

    (venv) $ git clone https://github.com/Sujit-O/pykg2vec.git
    (venv) $ cd pykg2vec
    (venv) $ git checkout development
    (venv) $ python setup.py install

.. _GitHub: https://github.com/Sujit-O/pykg2vec/pulls
.. _Tensorflow: https://www.tensorflow.org/install
