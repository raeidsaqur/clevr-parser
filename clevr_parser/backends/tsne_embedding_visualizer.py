#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tsne_embedding_visualizer.py
# Author: Ameet Deshpande
# Email: asd@cs.princeton.edu
# Created on: 2020-05-17
#
# This file is part of CLEVR-PARSER
# Distributed under terms of the MIT License
# https://github.com/raeidsaqur/clevr-parser

from ..embedding_visualizer import EmbeddingVisualizer, get_default_embedding_visualizer
from .backend import EmbeddingVisualizerBackend
from typing import List, Dict, Tuple, Sequence
import logging
logger = logging.getLogger(__name__)
import copy

# Imports for t-SNE
import numpy as np
from sklearn.manifold import TSNE

# Imports for clustering
from sklearn.cluster import KMeans

try:
    import matplotlib
    import matplotlib.pyplot as plt
    #import pygraphviz as pgv
    import networkx as nx
except ImportError as ie:
    logger.error(f"Install NetworkX: {ie.name}")

__all__ = ['tsneEmbeddingVisualizer']

@EmbeddingVisualizer.register_backend
class tsneEmbeddingVisualizer(EmbeddingVisualizerBackend):
    """
    Visualizer based on t-SNE  projections
    """
    __identifier__ = 'tsne'

    def __init__(self):
        super().__init__()

    @classmethod
    def draw_embeddings(cls, vectors: np.ndarray, show_clusters=False, *args, **kwargs):
        """
        :param vectors: 2-D matrix containing embeddings
        :param args: Additional args
        :param kwargs: Additoinal kw arguments like pos (for image_scene_graphs), ax_title etc.
        :return:
        """
        if show_clusters==True:
            return cls.draw_embeddings_tsne_cluster(vectors, *args, **kwargs)
        else:
            return cls.draw_embeddings_tsne(vectors, *args, **kwargs)

    @classmethod
    def get_tsne_embeddings(cls, vectors):
        """
        :param vectors: 2-D matrix containing embeddings
        :return:
        """
        # Instantiate object
        tsne = TSNE(n_components=2)

        # Fit and transform the data
        embedded_vectors = tsne.fit_transform(vectors)

        return embedded_vectors

    @classmethod
    def draw_embeddings_tsne(cls, vectors, labels=None, ax_title='t-SNE Visualization'):
        """
        :param vectors: 2-D matrix containing embeddings
        :param labels: The labels for the vectors
        :ax_title: 2-D matrix containing embeddings
        :return: The plot
        """
        # Get the TSNE vectors
        embedded_vectors = cls.get_tsne_embeddings(vectors)

        # Plot based on the presence of labels
        plt.scatter(embedded_vectors[:,0], embedded_vectors[:,1], c=labels)
        plt.title(ax_title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()

        return plt

    @classmethod
    def draw_embeddings_tsne_cluster(cls, vectors, n_clusters=3, clustering_method='kmeans', ax_title='t-SNE Cluster Visualization'):
        """
        :param vectors: 2-D matrix containing embeddings
        :param n_clusters: Number of cluster to divide the data into
        :param clustering_method: Choose from kmeans
        :return:
        """
        # Instantiate object
        if clustering_method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters)

        # Fit and transform the data
        labels = clustering.fit_predict(vectors)

        # Plot the embeddings
        plt = cls.draw_embeddings_tsne(vectors, labels)

        return plt