#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for controlling knowledge graph
"""


import shutil, tarfile, pickle
import urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np


class Triple(object):
    """The class defines the datastructure of the knowledge graph triples.

       Triple class is used to store the head, tail and relation triple in both its numerical id and
       string form. It also stores the dictonary of (head, relation)=[tail1, tail2,..] and
       (tail, relation)=[head1, head2, ...]

       Args:
          h (str or int): String or integer head entity.
          r (str or int): String or integer relation entity.
          t (str or int): String or integer tail entity.

       Attributes:
           h (int): Stores the head triple id.
           t (int): Stores the tail triple id.
           r (int): Stores the relation triple id.
           hr_t (dict): Stores the list of tails for head and relation pair.
           tr_h (dict): Stores the list of heads for tail and relation pair.

       Todo:
                * Move the module to config.
       Examples:
           >>> from pykg2vec.config.global_config import Triple
           >>> trip1 = Triple(2,3,5)
           >>> trip2 = Triple('Tokyo','isCapitalof','Japan')
    """
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
        """This function assigns the head, relation and tail.

            Args:
                h (int): Integer head entity.
                r (int): Integer relation entity.
                t (int): Integer tail entity.
        """
        self.h = h
        self.r = r
        self.t = t

    def set_strings(self, h, r, t):
        """This function assigns the head, relation and tail in string format.

            Args:
                h (str): String  head entity.
                r (str): String  relation entity.
                t (str): String  tail entity.

            Todo:
                * Assing the strings.
        """
        pass

    def set_hr_t(self, hr_t):
        """This function assigns the tails list for the given h,r pair.

            Args:
                hr_t (list): list of integer id of tails for given head, relation pair.
        """
        self.hr_t = hr_t

    def set_tr_h(self, tr_h):
        """This function assigns the head list for the given t,r pair.

            Args:
                tr_h (list): list of integer id of head for given tail, relation pair.
        """
        self.tr_h = tr_h


class KGMetaData(object):
    """The class store the metadata of the knowledge graph.

       Instance of KGMetaData is used later to build the algorithms based of number
       of entities and relation.

       Args:
            tot_entity (int):  Total number of combined head and tail entities present in knowledge graph.
            tot_relation(int): Total number of relations present in knowlege graph.
            tot_triple(int): Total number of head, relation and tail (triples) present in knowledge graph.
            tot_train_triples(int): Total number of training triples
            tot_test_triples(int): Total number of testing triple
            tot_valid_triples(int): Total number of validation triples

       Examples:
            >>> from pykg2vec.config.global_config import KGMetaData
            >>> kg_meta = KGMetaData(tot_triple =1000)

    """
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


def extract(tar_path, extract_path='.'):
    """This function extracts the tar file.

        Most of the knowledge graph dataset are donwloaded in a compressed
        tar format. This function is used to extract them

        Args:
            tar_path (str): Location of the tar folder.
            extract_path (str): Path where the files will be decompressed.

        Todo:
            * Move this module to utils!
    """
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])


class KnownDataset:
    """The class consists of modules to handle the known datasets.

       There are various known knowledge graph datasets used by the research
       community. These datasets maybe in different format. This module
       helps in parsing those known datasets for training and testing
       the algorithms.

       Args:
          name (str): Name of the datasets
          url (str): The full url where the dataset resides.
          prefix (str): The prefix of the dataset given the website.

       Attributes:
           dataset_home_path (object): Path object where the data will be downloaded
           root_oath (object): Path object for the specific dataset.

       Examples:
           >>> from pykg2vec.config.global_config import KnownDataset
           >>> name = "dLmL50"
           >>> url = "https://dl.dropboxusercontent.com/s/awoebno3wbgyrei/dLmL50.tgz?dl=0"
           >>> prefix = 'deeplearning_dataset_50arch-'
           >>> kgdata =  KnownDataset(name, url, prefix)
           >>> kgdata.download()
           >>> kgdata.extract()
           >>> kgdata.dump()

    """

    def __init__(self, name, url, prefix):

        self.name = name
        self.url = url
        self.prefix = prefix

        self.dataset_home_path = Path('..') / 'dataset'
        self.dataset_home_path.mkdir(parents=True, exist_ok=True)
        self.dataset_home_path = self.dataset_home_path.resolve()
        self.root_path = self.dataset_home_path / self.name
        self.tar = self.root_path / ('%s.tgz' % self.name)

        if not self.root_path.exists():
            self.download()
            self.extract()

        if self.name == 'WN18':
            self.dataset_path = self.root_path / 'wordnet-mlj12'
        elif self.name == 'YAGO3_10' or self.name == 'WN18RR':
            self.dataset_path = self.root_path
        else:
            self.dataset_path = self.root_path / self.name

        self.data_paths = {
            'train': self.dataset_path / ('%strain.txt'%self.prefix),
            'test': self.dataset_path / ('%stest.txt'%self.prefix),
            'valid': self.dataset_path / ('%svalid.txt'%self.prefix)
        }

        self.cache_triplet_paths = {
            'train': self.dataset_path / 'triplets_train.pkl',
            'test': self.dataset_path / 'triplets_test.pkl',
            'valid': self.dataset_path / 'triplets_valid.pkl'
        }

        self.cache_metadata_path = self.dataset_path / 'metadata.pkl'
        self.cache_hr_t_path = self.dataset_path / 'hr_t.pkl'
        self.cache_tr_h_path = self.dataset_path / 'tr_h.pkl'
        self.cache_idx2entity_path = self.dataset_path / 'idx2entity.pkl'
        self.cache_idx2relation_path = self.dataset_path / 'idx2relation.pkl'
        self.cache_entity2idx_path = self.dataset_path / 'entity2idx.pkl'
        self.cache_relation2idx_path = self.dataset_path / 'relation2idx.pkl'


    def download(self):
        ''' Downloads the given dataset from url'''
        print("Downloading the dataset %s" % self.name)

        self.root_path.mkdir()
        with urllib.request.urlopen(self.url) as response, open(str(self.tar), 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    def extract(self):
        ''' Extract the downloaded tar under the folder with the given dataset name'''
        print("Extracting the downloaded dataset from %s to %s" % (self.tar, self.root_path))

        try:
            extract(str(self.tar), str(self.root_path))
        except Exception as e:
            print("Could not extract the tgz file!")
            print(type(e), e.args)

    def read_metadata(self):
        ''' Reads the metadata of the knowledge graph if available'''
        with open(str(self.cache_metadata_path), 'rb') as f:
            meta = pickle.load(f)
            return meta

    def is_meta_cache_exists(self):
        ''' Checks if the metadata of the knowledge graph if available'''
        return self.cache_metadata_path.exists()

    def dump(self):
        ''' Displays all the metadata of the knowledge graph'''
        for key, value in self.__dict__.items():
            print(key, value)


class FreebaseFB15k(KnownDataset):
    """This data structure defines the necessary information for downloading Freebase dataset.

        FreebaseFB15k module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "FB15k"
        url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
        prefix = "freebase_mtr100_mte100-"

        KnownDataset.__init__(self, name, url, prefix)


class DeepLearning50a(KnownDataset):
    """This data structure defines the necessary information for downloading DeepLearning50a dataset.

        DeepLearning50a module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "dLmL50"
        url = "https://dl.dropboxusercontent.com/s/awoebno3wbgyrei/dLmL50.tgz?dl=0"
        prefix = 'deeplearning_dataset_50arch-'

        KnownDataset.__init__(self, name, url, prefix)


class WordNet18(KnownDataset):
    """This data structure defines the necessary information for downloading WordNet18 dataset.

        WordNet18 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "WN18"
        url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz"
        prefix = 'wordnet-mlj12-'

        KnownDataset.__init__(self, name, url, prefix)


class WordNet18_RR(KnownDataset):
    """This data structure defines the necessary information for downloading WordNet18_RR dataset.

        WordNet18_RR module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "WN18RR"
        url = "https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class YAGO3_10(KnownDataset):
    """This data structure defines the necessary information for downloading YAGO3_10 dataset.

        YAGO3_10 module inherits the KnownDataset class for processing
        the knowledge graph dataset.

        Attributes:
            name (str): Name of the datasets
            url (str): The full url where the dataset resides.
            prefix (str): The prefix of the dataset given the website.

    """
    def __init__(self):
        name = "YAGO3_10"
        url = "https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz"
        prefix = ''

        KnownDataset.__init__(self, name, url, prefix)


class UserDefinedDataset(object):
    """The class consists of modules to handle the user defined datasets.

      User may define their own datasets to be processed with the
      pykg2vec library.

      Args:
         name (str): Name of the datasets

      Attributes:
          dataset_home_path (object): Path object where the data will be downloaded
          root_oath (object): Path object for the specific dataset.

   """
    def __init__(self, name):
        self.name = name

        self.dataset_home_path = Path('..') / 'dataset'
        self.dataset_home_path.mkdir(parents=True, exist_ok=True)
        self.dataset_home_path = self.dataset_home_path.resolve()
        self.root_path = self.dataset_home_path / name

        if not self.root_path.exists():
            raise NotImplementedError("%s user defined dataset not found!" % self.root_path)

        train_file = self.root_path / (name + '-train.txt')
        test_file = self.root_path / (name + '-test.txt')
        valid_file = self.root_path / (name + '-valid.txt')

        if not train_file.exists():
            raise NotImplementedError("%s training file not found!" % train_file)
        if not test_file.exists():
            raise NotImplementedError("%s test file not found!" % test_file)
        if not test_file.exists():
            raise NotImplementedError("%s validation file not found!" % valid_file)

        self.data_paths = {
            'train': self.root_path / (name + '-train.txt'),
            'test': self.root_path / (name + '-test.txt'),
            'valid': self.root_path / (name + '-valid.txt')
        }

        self.cache_triplet_paths = {
            'train': self.root_path / 'triplets_train.pkl',
            'test': self.root_path / 'triplets_test.pkl',
            'valid': self.root_path / 'triplets_valid.pkl'
        }

        self.cache_metadata_path = self.root_path / 'metadata.pkl'
        self.cache_hr_t_path = self.root_path / 'hr_t.pkl'
        self.cache_tr_h_path = self.root_path / 'tr_h.pkl'
        self.cache_idx2entity_path = self.root_path / 'idx2entity.pkl'
        self.cache_idx2relation_path = self.root_path / 'idx2relation.pkl'
        self.cache_entity2idx_path = self.root_path / 'entity2idx.pkl'
        self.cache_relation2idx_path = self.root_path / 'relation2idx.pkl'

    def is_meta_cache_exists(self):
        """ Checks if the metadata has been cached"""
        return self.cache_metadata_path.exists()

    def read_metadata(self):
        """ Reads the metadata of the user defined dataset"""
        with open(str(self.cache_metadata_path), 'rb') as f:
            meta = pickle.load(f)
            return meta

    def dump(self):
        """ Prints the metadata of the user-defined dataset."""
        for key, value in self.__dict__.items():
            print(key, value)


class KnowledgeGraph(object):
    """The class is the main module that handles the knowledge graph.

      KnowledgeGraph is responsible for downloading, parsing, processing and preparing
      the training, testing and validation dataset.

      Args:
         dataset_name (str): Name of the datasets
         negative_sample (str): Sampling technique to be used for generating negative triples (bern or uniform).

      Attributes:
        dataset_name (str): The name of the dataset. 
        dataset (object): The dataset object isntance.
        negative_sample (str): negative_sample
        triplets (dict): dictionary with three list of training, testing and validation triples.
        relations (list):list of all the relations.
        entities (list): List of all the entities.
        entity2idx (dict): Dictionary for mapping string name of entities to unique numerical id.
        idx2entity (dict): Dictionary for mapping the id to string.
        relation2idx (dict): Dictionary for mapping the id to string.
        idx2relation (dict): Dictionary for mapping the id to string.
        hr_t (dict):  Dictionary with set as a default key and list as values.
        tr_h (dict):  Dictionary with set as a default key and list as values.
        hr_t_train (dict):  Dictionary with set as a default key and list as values.
        tr_h_train (dict):  Dictionary with set as a default key and list as values.
        relation_property (list): list storing the entities tied to a specific relation.
        kg_meta (object): Object storing the statistics metadata of the dataset.

      Examples:
          >>> from pykg2vec.config.global_config import KnowledgeGraph
          >>> knowledge_graph = KnowledgeGraph(dataset='Freebase15k', negative_sample='uniform')
          >>> knowledge_graph.prepare_data()
   """
    def __init__(self, dataset='Freebase15k', negative_sample='uniform'):
        
        self.dataset_name = dataset

        if dataset.lower() == 'freebase15k':
            self.dataset = FreebaseFB15k()
        elif dataset.lower() == 'deeplearning50a':
            self.dataset = DeepLearning50a()
        elif dataset.lower() == 'wordnet18':
            self.dataset = WordNet18()
        elif dataset.lower() == 'wordnet18_rr':
            self.dataset = WordNet18_RR()
        elif dataset.lower() == 'yago3_10':
            self.dataset = YAGO3_10()
        else:
            # if the dataset does not match with existing one, check if it exists in user's local space.
            # if it still can't find corresponding folder, raise exception in UserDefinedDataset.__init__()
            self.dataset = UserDefinedDataset(dataset)

        self.negative_sample = negative_sample

        self.triplets = {'train': [], 'test': [], 'valid': []}

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

    def force_prepare_data(self):
        if self.dataset.is_meta_cache_exists():
            self.dataset.cache_metadata_path.unlink()
        self.prepare_data()
        
    def prepare_data(self):
        """Function to prepare the dataset"""
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
        self.kg_meta.tot_entity = len(self.entities)
        self.kg_meta.tot_valid_triples = len(self.triplets['valid'])
        self.kg_meta.tot_test_triples = len(self.triplets['test'])
        self.kg_meta.tot_train_triples = len(self.triplets['train'])
        self.kg_meta.tot_triple = self.kg_meta.tot_valid_triples + \
                                  self.kg_meta.tot_test_triples + \
                                  self.kg_meta.tot_train_triples

        self.cache_data()

    def cache_data(self):
        """Function to cache the prepared dataset in the memory"""
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
        with open(str(self.dataset.cache_relation2idx_path), 'wb') as f:
            pickle.dump(self.entity2idx, f)
        with open(str(self.dataset.cache_entity2idx_path), 'wb') as f:
            pickle.dump(self.relation2idx, f)

    def read_cache_data(self, key):
        """Function to read the cached dataset from the memory"""
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

        elif key == 'entity2idx':
            with open(str(self.dataset.cache_entity2idx_path), 'rb') as f:
                entity2idx = pickle.load(f)

                return entity2idx

        elif key == 'relation2idx':
            with open(str(self.dataset.cache_relation2idx_path), 'rb') as f:
                relation2idx = pickle.load(f)

                return relation2idx

    def is_cache_exists(self):
        """Function to check if the dataset is cached in the memory"""
        return self.dataset.is_meta_cache_exists()

    def read_triplets(self, set_type):
        '''
            read triplets from txt files in dataset folder.
            (in string format)
        '''
        triplets = self.triplets[set_type]

        if len(triplets) == 0:
            with open(str(self.dataset.data_paths[set_type]), 'r', encoding='utf-8') as file:
                for line in file.readlines():
                    s, p, o = line.split('\t')
                    triplets.append(Triple(s.strip(), p.strip(), o.strip()))

        return triplets

    def read_entities(self):
        """ Function to read the entities. """
        if len(self.entities) == 0:
            entities = set()

            all_triplets = self.read_triplets('train') + \
                           self.read_triplets('valid') + \
                           self.read_triplets('test')

            for triplet in all_triplets:
                entities.add(triplet.h_string)
                entities.add(triplet.t_string)

            self.entities = np.sort(list(entities))

        return self.entities

    def read_relations(self):
        """ Function to read the relations. """
        if len(self.relations) == 0:
            relations = set()

            all_triplets = self.read_triplets('train') + \
                           self.read_triplets('valid') + \
                           self.read_triplets('test')

            for triplet in all_triplets:
                relations.add(triplet.r_string)

            self.relations = np.sort(list(relations))

        return self.relations

    def read_mappings(self):
        """ Function to generate the mapping from string name to integer ids. """
        self.entity2idx = {v: k for k, v in enumerate(self.read_entities())}  ##
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        self.relation2idx = {v: k for k, v in enumerate(self.read_relations())}  ##
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}

    def read_triple_ids(self, set_type):
        """ Function to read the triple idx.

            Args:
                set_type (str): Type of data, eithe train, test or valid.
        """
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
        """ Function to read the list of tails for the given head and relation pair. """
        for set_type in self.triplets:
            triplets = self.triplets[set_type]

            for t in triplets:
                self.hr_t[(t.h, t.r)].add(t.t)

        return self.hr_t

    def read_tr_h(self):
        """ Function to read the list of heads for the given tail and relation pair. """
        for set_type in self.triplets:
            triplets = self.triplets[set_type]

            for t in triplets:
                self.tr_h[(t.t, t.r)].add(t.h)

        return self.tr_h

    def read_hr_t_train(self):
        """ Function to read the list of tails for the given head and relation pair for the training set. """
        triplets = self.triplets['train']

        for t in triplets:
            self.hr_t_train[(t.h, t.r)].add(t.t)

        return self.hr_t_train

    def read_tr_h_train(self):
        """ Function to read the list of heads for the given tail and relation pair for the training set. """
        triplets = self.triplets['train']

        for t in triplets:
            self.tr_h_train[(t.t, t.r)].add(t.h)

        return self.tr_h_train

    def read_hr_tr_train(self):
        """ Function to read the list of heads for the given tail and relation pair
        and list of heads for the given tail and relation pair for the training set. """
        for t in self.triplets['train']:
            t.set_hr_t(self.hr_t_train[(t.h, t.r)])
            t.set_tr_h(self.tr_h_train[(t.t, t.r)])

        return self.triplets['train']

    def read_relation_property(self):
        """ Function to read the relation property.

         Returns:
             list: Returns the list of relation property.
         """
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
    def dump(self):
        """ Function to dump statistic information of a dataset """
        ''' dump key information'''
        print("\n----------Triple Stats for %s----------------" % self.dataset_name)
        print("Total Training Triples   :", len(self.triplets['train']))
        print("Total Testing Triples    :", len(self.triplets['test']))
        print("Total validation Triples :", len(self.triplets['valid']))
        print("Total Entities           :", self.kg_meta.tot_entity)
        print("Total Relations          :", self.kg_meta.tot_relation)
        print("---------------------------------------------")