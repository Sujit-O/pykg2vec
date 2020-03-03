#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Graph Meta Class
====================================
It provides Abstract class for the Knowledge graph models.
"""

from abc import ABCMeta, abstractmethod
import tensorflow as tf


class ModelMeta:
    """ Meta Class for knowledge graph embedding algorithms"""

    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialize and create the model to be trained and inferred"""
        pass

    @abstractmethod
    def def_inputs(self):
        """Function to define the inputs for the model"""
        pass

    @abstractmethod
    def def_parameters(self):
        """Function to define the parameters for the model"""
        pass

    @abstractmethod
    def def_loss(self):
        """Function to define how loss is calculated in the model"""
        pass

    @abstractmethod
    def embed(self,h, r, t):
        """Function to get the embedding value"""
        pass

    @abstractmethod
    def get_embed(self,h, r, t):
        """Function to get the embedding value in numpy"""
        pass

    @abstractmethod
    def get_proj_embed(self,h, r, t):
        """Function to get the projected embedding value"""
        pass

    def pairwise_margin_loss(self, score_positive, score_negative):
        ''' pairwise margin loss function 
            pairwise margin based ranking loss is defined as 

            for all positive samples:
                for all negative samples: 
                    loss += [margin + f(positive samples) - f(negative samples))]+
            
            In this method, the dimensions of score_positive and score_negative are assume to be equal. 
                => [b], where [b] is the number of batch.
        '''

        loss = tf.maximum(score_positive + self.config.margin - score_negative, 0)

        return tf.reduce_sum(loss)

    def pointwise_logistic_loss(self, score_positve, score_negative):
        ''' pointwise logistic loss function 
            pointwise logistic loss is defined as follows, 

            for all samples:
                if positive sample: loss -= f(positive samples) 
                if negative sample: loss += f(negative samples)                  
            
            In this method, the dimensions of score_negative should a multiple of score_positive. 
                => [b] or k*[b], where [b] is the number of batch.
        '''
        loss = tf.concat([-1*score_positve, score_negative], axis=0)
        loss = tf.nn.softplus(loss)

        return tf.reduce_sum(loss)


class TrainerMeta:
    """ Meta Class for Trainer Module"""
    __metaclass__ = ABCMeta

    def __init__(self):
        """Initializing and create the model to be trained and inferred"""
        pass

    @abstractmethod
    def build_model(self):
        """function to compile the model"""
        pass

    @abstractmethod
    def train_model(self):
        """function to train the model"""
        pass

    @abstractmethod
    def save_model(self, sess):
        """function to save the model"""
        pass

    @abstractmethod
    def load_model(self, sess):
        """function to load the model"""
        pass


class VisualizationMeta:
    """ Meta Class for Visualization Module"""
    __metaclass__ = ABCMeta
    
    def __init__(self):
        """Initializing and create the model to be trained and inferred"""
        pass

    @abstractmethod
    def display(self):
        """function to display embedding"""
        pass

    @abstractmethod
    def summary(self):
        """function to print the summary"""
        pass


class EvaluationMeta:
    """ Meta Class for Evaluation Module"""
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def relation_prediction(self):
        """Function for evaluating link prediction"""
        pass

    @abstractmethod
    def entity_classification(self):
        """Function for evaluating entity classification"""
        pass

    @abstractmethod
    def relation_classification(self):
        """Function for evaluating relation classification"""
        pass

    @abstractmethod
    def triple_classification(self):
        """Function for evaluating triple classificaiton"""
        pass

    @abstractmethod
    def entity_completion(self):
        """Function for evaluating entity completion"""
        pass


class InferenceMeta:
    """ Meta Class for inference based on distance measure"""
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def def_parameters(self):
        """Function to define the parameters for the model"""
        pass

    def dissimilarity(self, h, r, t):
        """Function to calculate dissimilarity measure in embedding space."""

    def infer_tails(self, h, r, topk):
        """Function to infer top k tails for given head and relation.

        Args:
            h (int): Head entities ids.
            r (int): Relation ids of the triple.
            topk (int): Top K values to infer.

        Returns:
            Tensors: Returns the list of tails tensor.
        """
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        head_vec = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        rel_vec = tf.nn.embedding_lookup(norm_rel_embeddings, r)

        score_tail = self.dissimilarity(head_vec, rel_vec, norm_ent_embeddings)
        _, tails = tf.nn.top_k(-score_tail, k=topk)

        return tails

    def infer_heads(self, r, t, topk):
        """Function to infer top k head for given relation and tail.

        Args:
            t (int): tail entities ids.
            r (int): Relation ids of the triple.
            topk (int): Top K values to infer.

        Returns:
            Tensors: Returns the list of heads tensor.
        """
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        tail_vec = tf.nn.embedding_lookup(norm_ent_embeddings, t)
        rel_vec = tf.nn.embedding_lookup(norm_rel_embeddings, r)

        score_head = self.dissimilarity(norm_ent_embeddings, rel_vec, tail_vec)
        _, heads = tf.nn.top_k(-score_head, k=topk)

        return heads

    def infer_rels(self, h, t, topk):
        """Function to infer top k relations for given head and tail.

        Args:
            h (int): Head entities ids.
            t (int): Tail entities ids.
            topk (int): Top K values to infer.

        Returns:
            Tensors: Returns the list of rels tensor.
        """
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        head_vec = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        tail_vec = tf.nn.embedding_lookup(norm_ent_embeddings, t)

        score_rel = self.dissimilarity(head_vec, norm_rel_embeddings, tail_vec)
        _, rels = tf.nn.top_k(-score_rel, k=topk)

        return rels

