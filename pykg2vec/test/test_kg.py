from pykg2vec.utils.kgcontroller import KnowledgeGraph


def test_fb15k():
    """Function to test the the knowledge graph parse for Freebase."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

def test_dl50a():
    """Function to test the the knowledge graph parse for Deep learning knowledge base."""
    knowledge_graph = KnowledgeGraph(dataset="deeplearning50a", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

def test_wn18():
    """Function to test the the knowledge graph parse for Wordnet dataset."""
    knowledge_graph = KnowledgeGraph(dataset="wordnet18", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

def test_wn18rr():
    """Function to test the the knowledge graph parse for Wordnet Dataset."""
    knowledge_graph = KnowledgeGraph(dataset="wordnet18_rr", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

def test_yago():
    """Function to test the the knowledge graph parse for Yago Dataset."""
    knowledge_graph = KnowledgeGraph(dataset="yago3_10", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()