from pykg2vec.utils.KGPipeline import KGPipeline

def main():
    """Function to test the KGPipeline function."""
    kg_pipeline = KGPipeline(model="transe", dataset ="Freebase15k", debug=True)
    kg_pipeline.tune()
    kg_pipeline.test()

if __name__ == "__main__":
    main()
