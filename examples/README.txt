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

- To experiment with your own hyperparameters, tweak the values inside ./examples/custom_hp.yaml or create your own files. ::

    $ python train.py -exp True -mn TransE -ds fb15k -hpf ./examples/custom_hp.yaml

- YAML formatting example (The file is also provided in /examples folder named custom_hp.yaml): ::

    model_name: "TransE"
    datasets:
      - dataset: "freebase15k"
        parameters:
          learning_rate: 0.01
          l1_flag: True
          hidden_size: 50
          batch_size: 128
          epochs: 1000
          margin: 1.00
          optimizer: "sgd"
          sampling: "bern"
          neg_rate: 1
    
- NB: To make sure the loaded hyperparameters will be actually used for training, you need to pass in the same model_name value via -mn and the same dataset value via -ds as already specified in the YAML file from where those hyperparameters are originated.

====

**Use Your Own Search Spaces**

- To tune a model with your own search space, tweak the values inside ./examples/custom_ss.yaml or create your own files. ::

    $ python tune_model.py -exp True -mn TransE -ds fb15k -ssf ./examples/custom_ss.yaml

- YAML formatting example (THe file is also provided in /examples folder named custom_ss.yaml): ::

    model_name: "TransE"
    dataset: "freebase15k"
    search_space:
        learning_rate:
            min: 0.00001
            max: 0.1
        l1_flag:
            - True
            - False
        hidden_size:
            min: 8
            max: 256
        batch_size:
            min: 8
            max: 4096
        margin:
            min: 0.0
            max: 10.0
        optimizer:
            - "adam"
            - "sgd"
            - "rms"
        epochs:
            - 10

- NB: To make sure the loaded search space will be actually used for tune_model.py, you need to pass in the same model_name value via -mn and the same dataset value via -ds as already specified in the YAML file that aligned with the parameters included in yaml.


====

**Use Your Own Dataset**

To create and use your own dataset, these steps are required:

1. Store all of triples in a text-format with each line as below, using tab space ("\t") to seperate entities and relations.::

    head\trelation\ttail

2. For the text file, separate it into three files according to your reference give names as follows, ::

    [name]-train.txt, [name]-valid.txt, [name]-test.txt

3. For those three files, create a folder [path_storing_text_files] to include them.
4. Create a new custom hyperparameter YAML file (detailed in "Use Your Own Hyperparameters"). For example, ::

    model_name: "TransE"
    datasets:
      - dataset: "[name]"
        parameters:
          learning_rate: 0.01
          l1_flag: True
          hidden_size: 50
          batch_size: 128
          epochs: 1000
          margin: 1.00
          optimizer: "sgd"
          sampling: "bern"
          neg_rate: 1

5. Once finished, you then can use your own dataset to train a KGE model or tune its hyperparameters using commands:::

    $ python train.py -mn TransE -ds [name] -dsp [path_storing_text_files] -hpf [path_to_hyperparameter_yaml]
    $ python tune_model.py -mn TransE -ds [name] -dsp [path_storing_text_files] -hpf [path_to_hyperparameter_yaml]


.. _LinkToEx: https://github.com/Sujit-O/pykg2vec/tree/master/examples
__ LinkToEx_

.. _train.py: train.html
.. _tune_model.py: tune_model.html
.. _inference.py: inference.html
.. _experiment.py: experiment.html
.. _kgpipeline.py: kgpipeline.html