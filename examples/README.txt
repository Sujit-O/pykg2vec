.. _general_examples:

Programming Examples
=====================================

We developed `several programming examples`__ for users to start working with pykg2vec, including:

- Work with one KGE method (train.py_)
- Automatic Hyperparameter Discovery (tune_model.py_)
- Inference task for one KGE method (inference.py_)
- Train multiple algorithms: (experiment.py_)
- Full pykg2vec pipeline: (kgpipeline.py_)

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