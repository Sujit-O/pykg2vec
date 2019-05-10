import shutil, tarfile, urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
import pickle

class Triple(object):
    
    def __init__(self, h, r, t):
        self.h = None
        self.r = None
        self.t = None
        self.h_string = None
        self.r_string = None
        self.t_string = None

        if type(h) is int and type(r) is int and type(t) is int:
            self.h = h
            self.r = r
            self.t = t
        
        else:
            self.h_string = h
            self.r_string = r
            self.t_string = t
        
        self.hr_t = None
        self.tr_h = None

    def set_ids(self, h, r, t):
        self.h = h 
        self.r = r
        self.t = t

    def set_strings(self, h, r, t):
        pass

    def set_hr_t(self, hr_t):
        self.hr_t = hr_t

    def set_tr_h(self, tr_h):
        self.tr_h = tr_h

class KGMetaData(object):
    def __init__(self, tot_entity=None,
                 tot_relation=None,
                 tot_triple=None,
                 tot_train_triples=None,
                 tot_test_triples=None,
                 tot_valid_triples=None):
        self.tot_triple = tot_triple
        self.tot_valid_triples = tot_valid_triples
        self.tot_test_triples = tot_test_triples
        self.tot_train_triples = tot_train_triples
        self.tot_relation = tot_relation
        self.tot_entity = tot_entity
        
# TODO: to be moved to utils
def extract(tar_path, extract_path='.'):
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])


class FreebaseFB15k(object):

    def __init__(self):
        self.name = "FB15k"
        self.url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
        self.dataset_home_path = Path('..') / 'dataset'
        self.dataset_home_path.mkdir(parents=True, exist_ok=True)
        self.dataset_home_path = self.dataset_home_path.resolve()
        self.root_path = self.dataset_home_path / 'Freebase'
        self.tar = self.root_path / 'FB15k.tgz'

        if not self.root_path.exists():
            self.download()
            self.extract()

        self.root_path = self.root_path / 'FB15k'
        self.downloaded_path    = self.root_path / 'freebase_mtr100_mte100-'

        self.data_paths = {
            'train': self.root_path / 'freebase_mtr100_mte100-train.txt',
            'test' : self.root_path / 'freebase_mtr100_mte100-test.txt',
            'valid': self.root_path / 'freebase_mtr100_mte100-valid.txt'
        }

        self.cache_path = self.root_path / 'all.pkl'
        self.cache_metadata_path = self.root_path / 'metadata.pkl'

        self.cache_triplet_paths = {
            'train': self.root_path / 'triplets_train.pkl',
            'test' : self.root_path / 'triplets_test.pkl',
            'valid': self.root_path / 'triplets_valid.pkl'
        }
        
        self.cache_hr_t_path = self.root_path / 'hr_t.pkl'
        self.cache_tr_h_path = self.root_path / 'tr_h.pkl'
        self.cache_idx2entity_path = self.root_path / 'idx2entity.pkl'
        self.cache_idx2relation_path = self.root_path / 'idx2relation.pkl'

       
    def download(self):
        ''' download Freebase 15k dataset from url'''
        print("Downloading the dataset %s" % self.name)

        self.root_path.mkdir()
        with urllib.request.urlopen(self.url) as response, open(str(self.tar), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    def extract(self):
        ''' extract the downloaded tar under Freebase 15k folder'''
        print("Extracting the downloaded dataset from %s to %s" % (self.tar, self.root_path))

        try:
            extract(str(self.tar), str(self.root_path))
        except Exception as e:
            print("Could not extract the tgz file!")
            print(type(e), e.args)

    def dump(self):
        for key, value in self.__dict__.items():
            print(key, value)

    def read_data(self):
        if self.cache_path.exists():
            with open(str(self.cache_path), 'rb') as f:
                knowledge_graph = pickle.load(f)

            return knowledge_graph
        return None
    
    def read_metadata(self):
        with open(str(self.cache_metadata_path), 'rb') as f:
            meta = pickle.load(f)

            return meta

    def is_meta_cache_exists(self):
        return self.cache_metadata_path.exists()

class DeepLearning50k(object):

    def __init__(self):
        self.name="dLmL50"
        self.url = "https://dl.dropboxusercontent.com/s/awoebno3wbgyrei/dLmL50.tgz?dl=0"
        self.dataset_home_path = Path('..')/'dataset'
        self.dataset_home_path.mkdir(parents=True, exist_ok=True)
        self.dataset_home_path = self.dataset_home_path.resolve()
        self.root_path = self.dataset_home_path/'DeepLearning'
        self.tar = self.root_path/'dLmL50.tgz'
        self.downloaded_path = self.root_path/'dLmL50'/'deeplearning_dataset_50arch-'
        self.prepared_data_path = self.root_path/'dLmL50'/'dLmL50_'
        self.entity2idx_path = self.root_path/'dLmL50'/'dLmL50_entity2idx.pkl'
        self.idx2entity_path = self.root_path/'dLmL50'/'dLmL50_idx2entity.pkl'
        self.relation2idx_path = self.root_path/'dLmL50'/'dLmL50_relation2idx.pkl'
        self.idx2relation_path = self.root_path/'dLmL50'/'dLmL50_idx2relation.pkl'

        if not self.root_path.exists():
            self.download()
            self.extract()

    def download(self):
        ''' Download dLmL50 dataset from url'''
        print("Downloading the dataset %s"%self.name)

        self.root_path.mkdir()
        with urllib.request.urlopen(self.url) as response, open(str(self.tar), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    def extract(self):
        ''' extract the downloaded tar under DeepLearning folder'''
        print("Extracting the downloaded dataset from %s to %s"% (self.tar, self.root_path))

        try:
            extract(str(self.tar), str(self.root_path))
        except Exception as e:
            print("Could not extract the tgz file!")
            print(type(e),e.args)

    def dump(self):
        for key, value in self.__dict__.items():
            print(key, value)


class KnowledgeGraph(object):

    def __init__(self, dataset='Freebase15k', negative_sample='uniform'):
        if dataset == 'Freebase15k':
            self.dataset = FreebaseFB15k()
        elif dataset == 'DeepLearning50k':
            self.dataset = DeepLearning50k()
        else:
            raise NotImplementedError("%s dataset config not found!" % dataset)

        self.negative_sample = negative_sample
        
        self.triplets = {'train': [], 'test' : [], 'valid': []}

        self.relations = [] 
        self.entities = []

        self.entity2idx = {} 
        self.idx2entity = {}
        self.relation2idx = {}
        self.idx2relation = {}

        self.hr_t = defaultdict(set)
        self.tr_h = defaultdict(set)

        self.hr_t_train = defaultdict(set)
        self.tr_h_train = defaultdict(set)

        self.relation_property = []

        if self.dataset.is_meta_cache_exists():
            self.kg_meta = self.dataset.read_metadata()
        else:
            self.kg_meta = KGMetaData()

    def prepare_data(self):
        if self.dataset.is_meta_cache_exists():
            return
            
        self.read_entities()
        self.read_relations()
        self.read_mappings()
        self.read_triple_ids('train')
        self.read_triple_ids('test')
        self.read_triple_ids('valid')
        self.read_hr_t()
        self.read_tr_h()
        self.read_hr_t_train()
        self.read_tr_h_train()
        
        self.read_hr_tr_train()

        if self.negative_sample == 'bern':
            self.read_relation_property()

        self.kg_meta.tot_relation = len(self.relations)
        self.kg_meta.tot_entity   = len(self.entities)
        self.kg_meta.tot_valid_triples = len(self.triplets['valid'])
        self.kg_meta.tot_test_triples  = len(self.triplets['test'])
        self.kg_meta.tot_train_triples = len(self.triplets['train'])
        self.kg_meta.tot_triple = self.kg_meta.tot_valid_triples + \
                                  self.kg_meta.tot_test_triples  + \
                                  self.kg_meta.tot_train_triples
        
    def cache_data(self):
        # with open(str(self.dataset.cache_path), 'wb') as f:
        #     pickle.dump(self, f)

        with open(str(self.dataset.cache_metadata_path), 'wb') as f:
            pickle.dump(self.kg_meta, f)
        with open(str(self.dataset.cache_triplet_paths['train']), 'wb') as f:
            pickle.dump(self.triplets['train'], f)
        with open(str(self.dataset.cache_triplet_paths['test']), 'wb') as f:
            pickle.dump(self.triplets['test'], f)
        with open(str(self.dataset.cache_triplet_paths['valid']), 'wb') as f:
            pickle.dump(self.triplets['valid'], f)
        with open(str(self.dataset.cache_hr_t_path), 'wb') as f:
            pickle.dump(self.hr_t, f)
        with open(str(self.dataset.cache_tr_h_path), 'wb') as f:
            pickle.dump(self.tr_h, f)
        with open(str(self.dataset.cache_idx2entity_path), 'wb') as f:
            pickle.dump(self.idx2entity, f)
        with open(str(self.dataset.cache_idx2relation_path), 'wb') as f:
            pickle.dump(self.idx2relation, f)

    def read_cache_data(self, key):
        if key == 'triplets_train':
            with open(str(self.dataset.cache_triplet_paths['train']), 'rb') as f:
                triplets = pickle.load(f)

                return triplets
        elif key == 'triplets_test':
            with open(str(self.dataset.cache_triplet_paths['test']), 'rb') as f:
                triplets = pickle.load(f)

                return triplets
        elif key == 'triplets_valid':
            with open(str(self.dataset.cache_triplet_paths['valid']), 'rb') as f:
                triplets = pickle.load(f)

                return triplets

        elif key == 'hr_t':
            with open(str(self.dataset.cache_hr_t_path), 'rb') as f:
                hr_t = pickle.load(f) 

                return hr_t

        elif key == 'tr_h':
            with open(str(self.dataset.cache_tr_h_path), 'rb') as f:
                tr_h = pickle.load(f) 

                return tr_h

        elif key == 'idx2entity':
            with open(str(self.dataset.cache_idx2entity_path), 'rb') as f:
                idx2entity = pickle.load(f)

                return idx2entity
        
        elif key == 'idx2relation':
            with open(str(self.dataset.cache_idx2relation_path), 'rb') as f:
                idx2relation = pickle.load(f)

                return idx2relation

    def is_cache_exists(self):
        return self.dataset.is_meta_cache_exists()

    def read_triplets(self, set_type):
        '''
            read triplets from txt files in dataset folder.
            (in string format)
        '''
        triplets = self.triplets[set_type]
        
        if len(triplets) == 0:
            with open(str(self.dataset.data_paths[set_type]), 'r') as file:
                for line in file.readlines():
                    s, p, o = line.split('\t')
                    triplets.append(Triple(s.strip(), p.strip(), o.strip()))

        return triplets
        
    def read_entities(self):
        ''' ensure '''
        if len(self.entities) == 0:
            entities = set()

            all_triplets = self.read_triplets('train') +\
                           self.read_triplets('valid') +\
                           self.read_triplets('test')

            for triplet in all_triplets:
                entities.add(triplet.h_string)
                entities.add(triplet.t_string)

            self.entities = np.sort(list(entities))

        return self.entities

    def read_relations(self):
        if len(self.relations) == 0:
            relations = set()

            all_triplets = self.read_triplets('train') +\
                           self.read_triplets('valid') +\
                           self.read_triplets('test')

            for triplet in all_triplets:
                relations.add(triplet.r_string)

            self.relations = np.sort(list(relations))

        return self.relations

    def read_mappings(self):
        self.entity2idx = {v: k for k, v in enumerate(self.read_entities())} ##
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        self.relation2idx = {v: k for k, v in enumerate(self.read_relations())} ##
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}

    def read_triple_ids(self, set_type):
        # assert entities can not be none
        # assert relations can not be none
        triplets = self.triplets[set_type]
        
        entity2idx = self.entity2idx
        relation2idx = self.relation2idx

        if len(triplets) != 0:
            for t in triplets:
                t.set_ids(entity2idx[t.h_string], relation2idx[t.r_string], entity2idx[t.t_string])

        return triplets

    def read_hr_t(self):
        for set_type in self.triplets:
            triplets = self.triplets[set_type]

            for t in triplets:
                self.hr_t[(t.h, t.r)].add(t.t)

        return self.hr_t

    def read_tr_h(self):
        for set_type in self.triplets:
            triplets = self.triplets[set_type]

            for t in triplets:
                self.tr_h[(t.t, t.r)].add(t.h)

        return self.tr_h

    def read_hr_t_train(self):
        triplets = self.triplets['train']

        for t in triplets:
            self.hr_t_train[(t.h, t.r)].add(t.t)

        return self.hr_t_train

    def read_tr_h_train(self):
        triplets = self.triplets['train']

        for t in triplets:
            self.tr_h_train[(t.t, t.r)].add(t.h)

        return self.tr_h_train
    
    def read_hr_tr_train(self):
        for t in self.triplets['train']:
            t.set_hr_t(self.hr_t_train[(t.h, t.r)])
            t.set_tr_h(self.tr_h_train[(t.t, t.r)])

        return self.triplets['train']

    def read_relation_property(self):
        relation_property_head = {x: [] for x in range(len(self.relations))}
        relation_property_tail = {x: [] for x in range(len(self.relations))}

        for t in self.triplets['train']:
            relation_property_head[t.r].append(t.h)
            relation_property_tail[t.r].append(t.t)

        self.relation_property = {x: (len(set(relation_property_tail[x]))) / ( \
                len(set(relation_property_head[x])) + len(set(relation_property_tail[x]))) \
                                  for x in relation_property_head.keys()}

        return self.relation_property

    ''' reserved for debugging '''
    # def dump(self):
    #     ''' dump key information'''
    #     print("\n----------Relation to Indexes--------------")
    #     pprint.pprint(self.relation2idx)
    #     print("---------------------------------------------")

    #     print("\n----------Relation to Indexes--------------")
    #     pprint.pprint(self.idx2relation)
    #     print("---------------------------------------------")

    #     print("\n----------Train Triple Stats---------------")
    #     print("Total Training Triples   :", len(self.train_triples_ids))
    #     print("Total Testing Triples    :", len(self.test_triples_ids))
    #     print("Total validation Triples :", len(self.validation_triples_ids))
    #     print("Total Entities           :", self.data_stats.tot_entity)
    #     print("Total Relations          :", self.data_stats.tot_relation)
    #     print("---------------------------------------------")

    # def dump_triples(self):
    #     '''dump all the triples'''
    #     for idx, triple in enumerate(self.train_triples):
    #         print(idx, triple.h, triple.r, triple.t)
    #     for idx, triple in enumerate(self.test_triples):
    #         print(idx, triple.h, triple.r, triple.t)
    #     for idx, triple in enumerate(self.validation_triples):
    #         print(idx, triple.h, triple.r, triple.t)

class GeneratorConfig(object):
    """Configuration for Generator

        Args:
          batch_size: int value for batch size
          loss_type : string 'distance' or 'entropy' to show
                     type of algorithm used
          data_path : path from where data is read and batch is
                     generated
          sampling  :  type of sampling used for generating negative
                     triples
          queue_size : size of the generator queue
          process_num: total number of processes for retrieving and
                       preparing the data
          total_data: total number of data stored in the disk, returned
                      by data_handler
    """

    def __init__(self, batch_size=128,
                 loss_type='entropy',
                 data_path=Path('..') / 'data',
                 sampling='uniform',
                 queue_size=50,
                 raw_queue_size=50,
                 processed_queue_size=50,
                 process_num=2,
                 data='train', 
                 algo ='ConvE',
                 neg_rate=2
                 ):
        self.neg_rate = neg_rate
        self.process_num = process_num
        self.raw_queue_size = raw_queue_size
        self.processed_queue_size = processed_queue_size
        self.algo = algo
        self.data = data
        self.queue_size = queue_size
        self.sampling = sampling
        self.data_path = data_path
        self.loss_type = loss_type
        self.batch_size = batch_size


def test_init_database():
    # global_config = KnowledgeGraph('Freebase15k')
    # global_config = KnowledgeGraph('Freebase100k')
    global_config = KnowledgeGraph('DeepLearning50k')

if __name__ == "__main__":
    test_init_database()
