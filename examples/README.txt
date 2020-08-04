.. _general_examples:

Programming Examples
=====================================

We developed `several programming examples`__ for users to start working with pykg2vec.
The examples are in ./examples folder. ::

    (pykg2vec) $ cd ./examples
    # train TransE using benchmark dataset fb15k
    (pykg2vec) $ python train.py -mn transe -ds fb15k
    # train and tune TransE using benchmark dataset fb15k
    (pykg2vec) $ python tune_model.py -mn TransE -ds fb15k

Please go through the examples for more advanced usages:

- Work with one KGE method (train.py_)
- Automatic Hyperparameter Discovery (tune_model.py_)
- Inference task for one KGE method (inference.py_)
- Train multiple algorithms: (experiment.py_)
- Full pykg2vec pipeline: (kgpipeline.py_)

====

**Use Your Own Hyperparameters**
To experiment with your own hyperparameters, tweak the values inside ./examples/hyperparams/custom.yaml or create your own files.

$ python train.py -exp True -mn TransE -ds fb15k -hpf ./examples/hyperparams/custom.yaml

For loading hyperparameters for multiple models altogether, you can pass in the path to the folder containing all your YAML configurations:

$ python train.py -exp True -mn TransE -ds fb15k -hpd ./examples/hyperparams

NB: To make sure the loaded hyperparameters will be actually used for training, you need to pass in the same model_name
value via -mn and the same dataset value via -ds as already specified in the YAML file from where those hyperparameters
are originated. If both -hpf and -hpd are present, the hyperparameters specified in the file following -hpf will take precedence.
====

**Use Your Own Dataset**

To create and use your own dataset, these steps are required:

1. Store all of triples in a text-format with each line as below, using tab space ("\t") to seperate entities and relations.::

    head\trelation\ttail

2. For the text file, separate it into three files according to your reference give names as follows, ::

    [name]-train.txt, [name]-valid.txt, [name]-test.txt

3. For those three files, create a folder [path_storing_text_files] to include them.
4. Once finished, you then can use your own dataset to train a KGE model using command:::

    $ python train.py -mn TransE -ds [name] -dsp [path_storing_text_files] 


.. _LinkToEx: https://github.com/Sujit-O/pykg2vec/tree/master/examples
__ LinkToEx_

.. _train.py: train.html
.. _tune_model.py: tune_model.html
.. _inference.py: inference.html
.. _experiment.py: experiment.html
.. _kgpipeline.py: kgpipeline.html