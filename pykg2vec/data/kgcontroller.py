#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for controlling knowledge graph
"""
import shutil
import pickle
import time
from collections import defaultdict
import numpy as np
from pykg2vec.utils.logger import Logger
from pykg2vec.data.datasets import (
    FreebaseFB15k,
    DeepLearning50a,
    WordNet18,
    WordNet18_RR,
    YAGO3_10,
    FreebaseFB15k_237,
    Kinship,
    Nations,
    UMLS,
    NELL_995,
    UserDefinedDataset,
)

class Triple:
    """ The class defines the datastructure of the knowledge graph triples.

        Triple class is used to store the head, tail and relation triple in both its numerical id and
        string form. It also stores the dictonary of (head, relation)=[tail1, tail2,..] and
        (tail, relation)=[head1, head2, ...]

        Args:
          h (str or int): String or integer head entity.
          r (str or int): String or integer relation entity.
          t (str or int): String or integer tail entity.

        Examples:
           >>> from pykg2vec.data.kgcontroller import Triple
           >>> trip1 = Triple(2,3,5)
           >>> trip2 = Triple('Tokyo','isCapitalof','Japan')
    """
    def __init__(self, h, r, t):
        self.h = h
        self.r = r
        self.t = t

    def set_ids(self, h, r, t):
        """ This function assigns the head, relation and tail.

            Args:
                h (int): Integer head entity.
                r (int): Integer relation entity.
                t (int): Integer tail entity.
        """
        self.h = h
        self.r = r
        self.t = t


class KGMetaData:
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
            >>> from pykg2vec.data.kgcontroller import KGMetaData
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


class KnowledgeGraph:
    """ The class is the main module that handles the knowledge graph.

        KnowledgeGraph is responsible for downloading, parsing, processing and preparing
        the training, testing and validation dataset.

        Args:
            dataset_name (str): Name of the datasets
            custom_dataset_path (str): The path to custom dataset.

        Attributes:
            dataset_name (str): The name of the dataset.
            dataset (object): The dataset object isntance.
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
            >>> from pykg2vec.data.kgcontroller import KnowledgeGraph
            >>> knowledge_graph = KnowledgeGraph(dataset='Freebase15k')
            >>> knowledge_graph.prepare_data()

    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, dataset='Freebase15k', custom_dataset_path=None):

        self.dataset_name = dataset

        if dataset.lower() == 'freebase15k' or dataset.lower() == 'fb15k':
            self.dataset = FreebaseFB15k()
        elif dataset.lower() == 'deeplearning50a' or dataset.lower() == 'dl50a':
            self.dataset = DeepLearning50a()
        elif dataset.lower() == 'wordnet18' or dataset.lower() == 'wn18':
            self.dataset = WordNet18()
        elif dataset.lower() == 'wordnet18_rr' or dataset.lower() == 'wn18_rr':
            self.dataset = WordNet18_RR()
        elif dataset.lower() == 'yago3_10' or dataset.lower() == 'yago':
            self.dataset = YAGO3_10()
        elif dataset.lower() == 'freebase15k_237' or dataset.lower() == 'fb15k_237':
            self.dataset = FreebaseFB15k_237()
        elif dataset.lower() == 'kinship' or dataset.lower() == 'ks':
            self.dataset = Kinship()
        elif dataset.lower() == 'nations':
            self.dataset = Nations()
        elif dataset.lower() == 'umls':
            self.dataset = UMLS()
        elif dataset.lower() == 'nell_995':
            self.dataset = NELL_995()
        elif custom_dataset_path is not None:
            # if the dataset does not match with existing one, check if it exists in user's local space.
            # if it still can't find corresponding folder, raise exception in UserDefinedDataset.__init__()

            self.dataset = UserDefinedDataset(dataset, custom_dataset_path)
        else:
            raise ValueError("Unknown dataset: %s" % dataset)

        # KG data structure stored in triplet format
        self.triplets = {'train': [], 'test': [], 'valid': []}
        self.triple_store = self.triplets

        # TODO: should also have graph-based data structure for a KG.
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

        self.hr_t_valid = defaultdict(set)
        self.tr_h_valid = defaultdict(set)

        self.relation_property = []

        if self.dataset.is_meta_cache_exists():
            self.kg_meta = self.dataset.read_metadata()
        else:
            self.kg_meta = KGMetaData()
            self.prepare_data()

    def force_prepare_data(self):
        shutil.rmtree(str(self.dataset.root_path), ignore_errors=True)

        time.sleep(1)

        self.__init__(dataset=self.dataset_name)

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
        self.read_hr_t_valid()
        self.read_tr_h_valid()
        self.read_relation_property()

        self.kg_meta.tot_relation = len(self.relations)
        self.kg_meta.tot_entity = len(self.entities)
        self.kg_meta.tot_valid_triples = len(self.triplets['valid'])
        self.kg_meta.tot_test_triples = len(self.triplets['test'])
        self.kg_meta.tot_train_triples = len(self.triplets['train'])
        self.kg_meta.tot_triple = self.kg_meta.tot_valid_triples + \
                                  self.kg_meta.tot_test_triples + \
                                  self.kg_meta.tot_train_triples

        self._cache_data()

    def _cache_data(self):
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
        with open(str(self.dataset.cache_hr_t_train_path), 'wb') as f:
            pickle.dump(self.hr_t_train, f)
        with open(str(self.dataset.cache_tr_h_train_path), 'wb') as f:
            pickle.dump(self.tr_h_train, f)
        with open(str(self.dataset.cache_idx2entity_path), 'wb') as f:
            pickle.dump(self.idx2entity, f)
        with open(str(self.dataset.cache_idx2relation_path), 'wb') as f:
            pickle.dump(self.idx2relation, f)
        with open(str(self.dataset.cache_relation2idx_path), 'wb') as f:
            pickle.dump(self.relation2idx, f)
        with open(str(self.dataset.cache_entity2idx_path), 'wb') as f:
            pickle.dump(self.entity2idx, f)
        with open(str(self.dataset.cache_relationproperty_path), 'wb') as f:
            pickle.dump(self.relation_property, f)

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

        elif key == 'hr_t_train':
            with open(str(self.dataset.cache_hr_t_train_path), 'rb') as f:
                hr_t_train = pickle.load(f)

                return hr_t_train

        elif key == 'tr_h_train':
            with open(str(self.dataset.cache_tr_h_train_path), 'rb') as f:
                tr_h_train = pickle.load(f)

                return tr_h_train

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

        elif key == 'relationproperty':
            with open(str(self.dataset.cache_relationproperty_path), 'rb') as f:
                relation_property = pickle.load(f)

                return relation_property
        else:
            raise ValueError('Unknown cache data key %s' % key)

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
                entities.add(triplet.h)
                entities.add(triplet.t)

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
                relations.add(triplet.r)

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
                t.set_ids(entity2idx[t.h], relation2idx[t.r], entity2idx[t.t])

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

    def read_hr_t_valid(self):
        """ Function to read the list of tails for the given head and relation pair for the valid set. """
        triplets = self.triplets['valid']

        for t in triplets:
            self.hr_t_valid[(t.h, t.r)].add(t.t)

        return self.hr_t_valid

    def read_tr_h_valid(self):
        """ Function to read the list of heads for the given tail and relation pair for the valid set. """
        triplets = self.triplets['valid']

        for t in triplets:
            self.tr_h_valid[(t.t, t.r)].add(t.h)

        return self.tr_h_valid

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

        self.relation_property = {}
        for x in relation_property_head.keys():
            value_up = len(set(relation_property_tail[x]))

            value_bot = len(set(relation_property_head[x])) + len(set(relation_property_tail[x]))

            if value_bot == 0:
                value = 0
            else:
                value = value_up / value_bot

            self.relation_property[x] = value

        return self.relation_property

    # reserved for debugging
    def dump(self):
        """ Function to dump statistic information of a dataset """
        # dump key information
        dump = [
            "",
            "----------Metadata Info for Dataset:%s----------------" % self.dataset_name,
            "Total Training Triples   :%s" % self.kg_meta.tot_train_triples,
            "Total Testing Triples    :%s" % self.kg_meta.tot_test_triples,
            "Total validation Triples :%s" % self.kg_meta.tot_valid_triples,
            "Total Entities           :%s" % self.kg_meta.tot_entity,
            "Total Relations          :%s" % self.kg_meta.tot_relation,
            "---------------------------------------------",
            "",
        ]
        self._logger.info("\n".join(dump))
        return dump
