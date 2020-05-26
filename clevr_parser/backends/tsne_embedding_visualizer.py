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
from sklearn.neighbors import NearestNeighbors

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
    def get_tsne_embeddings(cls, vectors, random_state=42):
        """
        :param vectors: 2-D matrix containing embeddings
        :return:
        """
        # Instantiate object
        tsne = TSNE(n_components=2, random_state=random_state)

        # Fit and transform the data
        embedded_vectors = tsne.fit_transform(vectors)

        return embedded_vectors

    @classmethod
    def draw_embeddings_tsne(cls, vectors, labels=None,
                             random_state=42,
                             ax_title='t-SNE Visualization'):
        """
        :param vectors: 2-D matrix containing embeddings
        :param labels: The labels for the vectors
        :ax_title: 2-D matrix containing embeddings
        :return: The plot
        """
        # Get the TSNE vectors
        embedded_vectors = cls.get_tsne_embeddings(vectors, random_state)

        # Plot based on the presence of labels
        plt.scatter(embedded_vectors[:,0], embedded_vectors[:,1], c=labels)
        plt.title(ax_title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()

        return plt

    @classmethod
    def draw_embeddings_tsne_cluster(cls, vectors, labels=None, n_clusters=3,
                                     clustering_method='kmeans',
                                     random_state=42,
                                     ax_title='t-SNE Cluster Visualization'):
        """
        :param vectors: 2-D matrix containing embeddings
        :param labels: The labels for the vectors
        :param n_clusters: Number of cluster to divide the data into
        :param clustering_method: Choose from kmeans
        :return:
        """
        # If true labels are provided, plot them
        if labels is not None:
            return cls.draw_embeddings_tsne(vectors, labels, ax_title=ax_title)
        else:
            # Instantiate object
            if clustering_method == 'kmeans':
                clustering = KMeans(n_clusters=n_clusters, random_state=42)

            # Fit and transform the data
            cluster_labels = clustering.fit_predict(vectors)

            # Plot the embeddings
            return cls.draw_embeddings_tsne(vectors, cluster_labels, ax_title=ax_title)

    @classmethod
    def get_nearest_neighbors(cls, vectors, n_neighbors=2, pivots=None):
        """
        :param vectors: 2-D matrix containing embeddings
        :param n_neighbors: Number of neighbors to return per data point
        :param pivots: List of indices. If none returns a dict containing neighbors only for those indices
        :return:
        """
        # Make sure the number of neighbors to return is less than or
        # equal to the total neighbors
        assert len(vectors) > n_neighbors, "Total neighbors is less than requested number of neighbors"

        # Ensure all indices are within range
        assert (pivots is None or np.all(np.array(pivots) < len(vectors))), "Illegal index requested"

        # Instantiate object
        NN = NearestNeighbors(n_neighbors=n_neighbors+1)

        # Fit to the data
        NN.fit(vectors)

        # Get the neighbors
        distances, indices = NN.kneighbors(vectors)

        # Return the nearest neighbors
        if pivots:
            neighbors = {}
            for idx in pivots:
                # Attach all neighbors (the first one is always identity)
                neighbors[idx] = indices[idx,1:]
            return neighbors
        else:
            neighbors = indices[:,1:]
            return neighbors         