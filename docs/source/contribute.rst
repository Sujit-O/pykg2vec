########################
Contribute to pykg2vec
########################

We feel humbled that you have decided to contribute to the pykg2vec repository.
Thank you! Please read the following guidelines to checkout how you can contribute.

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

* **Reporting Bugs**: Please use the issue Template_ to report bugs.
* **Suggesting Enhancements**: If you have any suggestion for enhancing any of the modules please send us an enhancement using the issue Template_ as well.
* **Adding Algorithm**: We are continually striving to add the state-of-the-art algorithms in the library. If you want to suggest adding any algorithm or add your algoirithm to the library, please follow the following steps:

    * Make sure the generator is able to produce the batches
    * Make sure to follow the class structure presented in pykg2vec/core/KGMeta.py

* **Adding Evaluation Metric**: We are always eager to add more evaluation metrics for link prediction, triple classification, and so on. You may create a new evaluation process in pykg2vec/utils/evaluation.py to add the metric.
* **Csource for Python Modules**: Although we use Tensorflow for running the main modules, there are many alforithms written in pure python. We invite you to contibute by converting the python source code to more efficient C or C++ codes.
* **Adding Dataset Source**: We encourage you to add your own dataset links. Currenlty the pykg2vec/config/global_config.py handles the datasets, how to extract them and generate the training, testing and validation triples.

.. _GitHub: https://github.com/Sujit-O/pykg2vec/pulls
.. _Template: https://github.com/Sujit-O/pykg2vec/blob/master/ISSUE_TEMPLATE.md
