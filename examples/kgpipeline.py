'''
=========================
Full Pykg2vec pipeline
=========================
In this example, we will demonstrate how the full
pykg2vec pipeline can be used. This pipeline includes tuning the 
algorithm's hyperparameter using training and validation set.
Then using the tuned hyper-parameter train the algorithm again on
combined training and validation data to finally get the results on 
the testing set. 
'''
# Author: Sujit Rokka Chhetri and Shiy Yuan Yu
# License: MIT

from pykg2vec.utils.KGPipeline import KGPipeline

def main():
    """Function to test the KGPipeline function."""
    kg_pipeline = KGPipeline(model="transe", dataset ="Freebase15k", debug=True)
    kg_pipeline.tune()
    kg_pipeline.test()

if __name__ == "__main__":
    main()
