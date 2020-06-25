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

====

We also attached the source code of Train.py below for your reference.
You can adjust to fit your usage.

'''
# Author: Sujit Rokka Chhetri and Shih Yuan Yu
# License: MIT

from pykg2vec.config.config import Importer, KGEArgParser
from pykg2vec.utils.kgcontroller import KnowledgeGraph
from pykg2vec.config.hyperparams import KGETuneArgParser
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer
from pykg2vec.utils.trainer import Trainer

def main():
    model_name = "transe"
    dataset = "Freebase15k"
        
    # Fuction to tune the hyper-parameters for the model 
    # using training and validation set.
    # getting the customized configurations from the command-line arguments.
    args = KGETuneArgParser().get_args([])
    args.model = model_name
    args.dataset_name = dataset
    # initializing bayesian optimizer and prepare data.
    bays_opt = BaysOptimizer(args=args)

    # perform the golden hyperparameter tuning. 
    bays_opt.optimize()
    best = bays_opt.return_best()



    # Function to evaluate final model on testing set while training the model using 
    # best hyper-paramters on merged training and validation set.
    args = KGEArgParser().get_args([])
    args.model = model_name
    args.dataset_name = dataset
    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args=args)
    
    # Update the config params with the golden hyperparameter
    for k,v in best.items():
        config.__dict__[k]=v
    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()

if __name__ == "__main__":
    main()
