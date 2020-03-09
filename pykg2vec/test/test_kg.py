import os, pytest
from pykg2vec.utils.kgcontroller import KnowledgeGraph

@pytest.mark.parametrize("dataset_name", ["freebase15k", "wordnet18", "wordnet18_rr", "yago3_10"])
@pytest.mark.parametrize("sampling", ["uniform", "bern"])
def test_benchmarks(dataset_name, sampling):
    """Function to test the the knowledge graph parse for Freebase."""
    print("testing downloading from online sources of benchmarks and KG controller's handling.")
    knowledge_graph = KnowledgeGraph(dataset=dataset_name, negative_sample=sampling)
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


def test_userdefined_dataset():
    custom_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource/custom_dataset")
    knowledge_graph = KnowledgeGraph(dataset="userdefineddataset", negative_sample="uniform", custom_dataset_path=custom_dataset_path)
    knowledge_graph.prepare_data()
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

    knowledge_graph.dataset.read_metadata()
    knowledge_graph.dataset.dump()

    assert knowledge_graph.kg_meta.tot_train_triples == 1
    assert knowledge_graph.kg_meta.tot_test_triples == 1
    assert knowledge_graph.kg_meta.tot_valid_triples == 1
    assert knowledge_graph.kg_meta.tot_entity == 6
    assert knowledge_graph.kg_meta.tot_relation == 3
