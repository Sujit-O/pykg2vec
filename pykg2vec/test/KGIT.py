import unittest

from pykg2vec.utils.kgcontroller import KnowledgeGraph


class Pykg2vecKGIT(unittest.TestCase):
    '''This test suite is for testing knowledge graph preparation'''
    
    def setUp(self):
        print('setup')
    
    def test_fb15k(self):
        knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
        knowledge_graph.force_prepare_data()
        knowledge_graph.dump()

    def test_dl50a(self):
        knowledge_graph = KnowledgeGraph(dataset="deeplearning50a", negative_sample="uniform")
        knowledge_graph.force_prepare_data()
        knowledge_graph.dump()

    def test_wn18(self):
        knowledge_graph = KnowledgeGraph(dataset="wordnet18", negative_sample="uniform")
        knowledge_graph.force_prepare_data()
        knowledge_graph.dump()

    def test_wn18rr(self):
        knowledge_graph = KnowledgeGraph(dataset="wordnet18_rr", negative_sample="uniform")
        knowledge_graph.force_prepare_data()
        knowledge_graph.dump()

    def test_yago(self):
        knowledge_graph = KnowledgeGraph(dataset="yago3_10", negative_sample="uniform")
        knowledge_graph.force_prepare_data()
        knowledge_graph.dump()

    def tearDown(self):
        print('teardown')


def suite():
    suite = unittest.TestSuite()

    suite.addTest(Pykg2vecKGIT('test_fb15k'))
    suite.addTest(Pykg2vecKGIT('test_dl50a'))
    suite.addTest(Pykg2vecKGIT('test_wn18'))
    suite.addTest(Pykg2vecKGIT('test_wn18rr'))
    suite.addTest(Pykg2vecKGIT('test_yago'))

    return suite

if __name__ == '__main__':
    """ Execute whole test case as a whole """
    runner = unittest.TextTestRunner()
    runner.run(suite())