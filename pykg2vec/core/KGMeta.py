#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Graph Meta Class
====================================
It provides Abstract class for the Knowledge graph models.
"""

from abc import ABCMeta, abstractmethod
import tensorflow as tf

class ModelMeta(tf.keras.Model):
    """ Meta Class for knowledge graph embedding algorithms"""

    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialize and create the model to be trained and inferred"""
        super(ModelMeta, self).__init__()

    @abstractmethod
    def def_parameters(self):
        """Function to define the parameters for the model"""
        pass

    @abstractmethod
    def get_loss(self):
        """Function to define how loss is calculated in the model"""
        pass

    @abstractmethod
    def embed(self,h, r, t):
        """Function to get the embedding value"""
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