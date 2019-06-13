import pytest
from pykg2vec.config.config import *
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.kgcontroller import KnowledgeGraph
import tensorflow as tf

@pytest.mark.skip(reason="This is a functional method.")
def testing_function(name, distance_measure=None, bilinear=None):
    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(name)
    config = config_def()
    
    config.epochs     = 1
    config.test_step  = 1
    config.test_num   = 10
    config.disp_result= False
    config.save_model = False

    if distance_measure is not None:
        config.distance_measure = distance_measure
    if bilinear is not None:
        config.bilinear = bilinear

    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=True)
    trainer.build_model()
    trainer.train_model()

    tf.reset_default_graph()

@pytest.mark.skip(reason="This is a functional method.")
def testing_function_with_args(name, distance_measure=None, bilinear=None):
    
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args([])
    
    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(name)
    config = config_def(args=args)
    
    config.epochs     = 1
    config.test_step  = 1
    config.test_num   = 10
    config.disp_result= False
    config.save_model = False

    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=True)
    trainer.build_model()
    trainer.train_model()

    tf.reset_default_graph()


def test_Complex():
    testing_function('complex')

def test_ConvE():
    testing_function('conve')
    
def test_DistMult():
    testing_function('distmult')

def test_KG2E_EL():
    testing_function('kg2e', distance_measure="expected_likelihood")

def test_KG2E_KL():
    testing_function('kg2e', distance_measure="kl_divergence")

def test_NTN():
    testing_function('ntn')

def test_ProjE():
    testing_function('proje_pointwise')

def test_RESCAL():
    testing_function('rescal')

def test_RotatE():
    testing_function('rotate')

def test_SLM():
    testing_function('slm')

def test_SMEL():
    testing_function('sme', bilinear=False)

def test_SMEB():
    testing_function('sme', bilinear=True)

def test_transE():
    testing_function('transe')

def test_transH():
    testing_function('transh')

def test_transR():
    testing_function('transr')

def test_TransD():
    testing_function('transd')

def test_TransM():
    testing_function('transm')

def test_HoLE():
    testing_function('hole')

def test_Tucker():
    testing_function('tucker')

def test_Complex_args():
    testing_function_with_args('complex')

def test_ConvE_args():
    testing_function_with_args('conve')
    
def test_DistMult_args():
    testing_function_with_args('distmult')

def test_KG2E_EL_args():
    testing_function_with_args('kg2e', distance_measure="expected_likelihood")

def test_KG2E_KL_args():
    testing_function_with_args('kg2e', distance_measure="kl_divergence")

def test_NTN_args():
    testing_function_with_args('ntn')

def test_ProjE_args():
    testing_function_with_args('proje_pointwise')

def test_RESCAL_args():
    testing_function_with_args('rescal')

def test_RotatE_args():
    testing_function_with_args('rotate')

def test_SLM_args():
    testing_function_with_args('slm')

def test_SMEL_args():
    testing_function_with_args('sme', bilinear=False)

def test_SMEB_args():
    testing_function_with_args('sme', bilinear=True)

def test_transE_args():
    testing_function_with_args('transe')

def test_transH_args():
    testing_function_with_args('transh')

def test_transR_args():
    testing_function_with_args('transr')

def test_TransD_args():
    testing_function_with_args('transd')

def test_TransM_args():
    testing_function_with_args('transm')

def test_HoLE_args():
    testing_function_with_args('hole')

def test_Tucker_args():
    testing_function_with_args('tucker')