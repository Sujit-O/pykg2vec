from pykg2vec.utils.kgcontroller import KnowledgeGraph
  
   
def test_fb15k():
    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

def test_dl50a():
    knowledge_graph = KnowledgeGraph(dataset="deeplearning50a", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

def test_wn18():
    knowledge_graph = KnowledgeGraph(dataset="wordnet18", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

def test_wn18rr():
    knowledge_graph = KnowledgeGraph(dataset="wordnet18_rr", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()

def test_yago():
    knowledge_graph = KnowledgeGraph(dataset="yago3_10", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    knowledge_graph.dump()