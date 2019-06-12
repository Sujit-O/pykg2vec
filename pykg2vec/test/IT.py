import unittest

from pykg2vec.config.config import *
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.kgcontroller import KnowledgeGraph
import tensorflow as tf

class Pykg2vecIT(unittest.TestCase):
    
    def setUp(self):
        print('setup')
       
    def test_function(self, name):
        knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
        knowledge_graph.prepare_data()

        # Extracting the corresponding model config and definition from Importer().
        config_def, model_def = Importer().import_model_config(name)
        config = config_def()
        
        config.test_step  = 1
        config.test_num   = 10
        config.disp_result= False
        config.save_model = False

        model = model_def(config)

        # Create, Compile and Train the model. While training, several evaluation will be performed.
        trainer = Trainer(model=model, debug=True)
        trainer.build_model()
        trainer.train_model()

    def test_Complex(self):
        self.test_function('complex')

    def test_ConvE(self):
        self.test_function('conve')
        
    def test_DistMult(self):
        self.test_function('distmult')

    def test_KG2E_EL(self):
        self.test_function('kg2e')
        # config = KG2EConfig(batch_size=512, epochs=1, distance_measure="expected_likelihood")
        # config.set_dataset("Freebase15k")
        
        # config.test_step = 1
        # config.test_num  = 10
        # config.gpu_fraction = 0.4
        # config.save_model = False
        # config.disp_result= False


        # model = KG2E(config)
        
        # trainer = Trainer(model=model, debug=True)
        # trainer.build_model()
        # trainer.train_model()
    
    def test_KG2E_KL(self):
        self.test_function('kg2e')

        # config = KG2EConfig(batch_size=512, epochs=1, distance_measure="kl_divergence")
        # config.set_dataset("Freebase15k")
        
        # config.test_step = 1
        # config.test_num  = 10
        # config.gpu_fraction = 0.4
        # config.save_model = False
        # config.disp_result= False

        # model = KG2E(config)
        
        # trainer = Trainer(model=model, debug=True)
        # trainer.build_model()
        # trainer.train_model()

    def test_NTN(self):
        self.test_function('ntn')
    
    def test_ProjE(self):
        self.test_function('proje_pointwise')

    def test_RESCAL(self):
        self.test_function('rescal')
    
    def test_RotatE(self):
        self.test_function('rotate')

    def test_SLM(self):
        self.test_function('slm')

    def test_SMEL(self):
        self.test_function('sme')

        # config = SMEConfig(batch_size=512, epochs=1, hidden_size=8)
        # config.set_dataset("Freebase15k")
        
        # config.test_step = 1
        # config.test_num  = 10
        # config.gpu_fraction = 0.4
        # config.save_model = True
        # config.disp_result= False
        # config.bilinear = False

        # model = SME(config)
        
        # trainer = Trainer(model=model, debug=True)
        # trainer.build_model()
        # trainer.train_model()

    def test_SMEB(self):
        self.test_function('sme')

        # config = SMEConfig(batch_size=512, epochs=1, hidden_size=8)
        # config.set_dataset("Freebase15k")
        
        # config.test_step = 1
        # config.test_num  = 10
        # config.gpu_fraction = 0.4
        # config.save_model = True
        # config.disp_result= False
        # config.bilinear = True

        # model = SME(config)
        
        # trainer = Trainer(model=model, debug=True)
        # trainer.build_model()
        # trainer.train_model()

    def test_transE(self):
        self.test_function('transe')
    
    def test_transH(self):
        self.test_function('transh')
    
    def test_transR(self):
        self.test_function('transr')
    
    def test_TransD(self):
        self.test_function('transd')

    def test_TransM(self):
        self.test_function('transm')

    def tearDown(self):
        print('teardown')
        tf.reset_default_graph()


def suite():
    suite = unittest.TestSuite()

    suite.addTest(Pykg2vecIT('test_Complex'))
    suite.addTest(Pykg2vecIT('test_ConvE'))
    suite.addTest(Pykg2vecIT('test_DistMult'))
    suite.addTest(Pykg2vecIT('test_KG2E_EL'))
    suite.addTest(Pykg2vecIT('test_KG2E_KL'))
    suite.addTest(Pykg2vecIT('test_NTN'))
    suite.addTest(Pykg2vecIT('test_ProjE'))
    suite.addTest(Pykg2vecIT('test_RESCAL'))
    suite.addTest(Pykg2vecIT('test_RotatE'))
    suite.addTest(Pykg2vecIT('test_SLM'))
    suite.addTest(Pykg2vecIT('test_SMEL'))
    suite.addTest(Pykg2vecIT('test_SMEB'))
    suite.addTest(Pykg2vecIT('test_transE'))
    suite.addTest(Pykg2vecIT('test_transH'))
    suite.addTest(Pykg2vecIT('test_transR'))
    suite.addTest(Pykg2vecIT('test_TransD'))
    suite.addTest(Pykg2vecIT('test_TransM'))
    
    return suite

if __name__ == '__main__':
    """ Execute whole test case as a whole """
    runner = unittest.TextTestRunner()
    runner.run(suite())
