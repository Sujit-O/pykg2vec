.. _general_examples:

Programming Examples
=====================================

To demonstrate the usage of pykg2vec, we developed several programming examples for your convenience. 

General-purpose and introductory examples for the `pykg2vec` library.


**Use Your Own Dataset**
To use your own dataset, these steps are required:

1. Store all of triples in a text-format with each line as below, using tab space ("\t") to seperate entities and relations.::
    
    head\trelation\ttail

2. For the text file, separate it into three files according to your reference give names as follows, ::

    [name]-train.txt, [name]-valid.txt, [name]-test.txt

3. For those three files, create a folder [path_storing_text_files] to include them.
4. Once finished, you then can use the custom dataset to train on a specific model using command:::

    $ python train.py -mn TransE -ds [name] -dsp [path_storing_text_files] 

