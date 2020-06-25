.. _general_examples:

Programming Examples
=====================================

To demonstrate the usage of pykg2vec, we developed several programming examples to show how to work with pykg2vec.

- Work with one KGE method: a short example to demonstrate how to train a model.
- Automatic Hyperparameter Discovery: shows how to tune a model and find golden Hyperparameter.
- Inference task for one KGE method: shows how interactive inference can be performed.
- Train multiple algorithms: shows how to train several models one after another.
- Full pykg2vec pipeline: shows how to work with a full pykg2vec pipeline,

====

**Use Your Own Dataset**
Besides using the datasets provided, you can also create your own dataset.
To create and use your own dataset, these steps are required:

1. Store all of triples in a text-format with each line as below, using tab space ("\t") to seperate entities and relations.::
    
    head\trelation\ttail

2. For the text file, separate it into three files according to your reference give names as follows, ::

    [name]-train.txt, [name]-valid.txt, [name]-test.txt

3. For those three files, create a folder [path_storing_text_files] to include them.
4. Once finished, you then can use the custom dataset to train on a specific model using command:::

    $ python train.py -mn TransE -ds [name] -dsp [path_storing_text_files] 

