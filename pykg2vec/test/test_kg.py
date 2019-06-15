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

def test_fb15k_manipulate():
    """Function to test the the knowledge graph parse for Freebase and basic operations."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="bern")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

    knowledge_graph.read_cache_data('triplets_train')
    knowledge_graph.read_cache_data('triplets_test')
    knowledge_graph.read_cache_data('triplets_valid')
    knowledge_graph.read_cache_data('hr_t')
    knowledge_graph.read_cache_data('tr_h')
    knowledge_graph.read_cache_data('idx2entity')
    knowledge_graph.read_cache_data('idx2relation')
    knowledge_graph.read_cache_data('entity2idx')
    knowledge_graph.read_cache_data('relation2idx')

def test_fb15k_meta():
    """Function to test the the knowledge graph parse for Freebase and basic operations."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="bern")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

    assert knowledge_graph.is_cache_exists()
    knowledge_graph.prepare_data()

    knowledge_graph.dataset.read_metadata()
    knowledge_graph.dataset.dump()


# def test_userdefined_dataset():

#     import os 
#     print()
#     knowledge_graph = KnowledgeGraph(dataset="userdefineddataset", negative_sample="uniform")
#     knowledge_graph.prepare_data()
#     knowledge_graph.dump()

#     knowledge_graph.dataset.read_metadata()
#     knowledge_graph.dataset.dump()
