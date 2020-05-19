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
    def draw_embeddings(cls, vectors: np.ndarray, *args, **kwargs):
        """
        :param vectors: 2-D matrix containing embeddings
        :param args: Additional args
        :param kwargs: Additoinal kw arguments like pos (for image_scene_graphs), ax_title etc.
        :return:
        """
        return cls.draw_embeddings_tsne(vectors, *args, **kwargs)

    @classmethod
    def draw_embeddings_tsne(cls, vectors):
        """
        :param vectors: 2-D matrix containing embeddings
        :return:
        """
        # Instantiate object
        tsne = TSNE(n_components=2)

        # Fit and transform the data
        embedded_vectors = tsne.fit_transform(vectors)

        # Plot the embeddings
        plt.scatter(embedded_vectors[:,0], embedded_vectors[:,1])
        plt.title("t-SNE Visualization")
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()

        return plt
