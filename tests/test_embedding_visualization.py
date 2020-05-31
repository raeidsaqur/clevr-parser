#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_embedding_visualization.py
# Author : Ameet Deshpande
# Email  : asd@cs.princeton.edu
# Date   : 05/17/2020
#
# This file is part of CLEVR Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

import pytest

np = pytest.importorskip('numpy')
from sklearn.datasets import make_blobs

import clevr_parser


@pytest.fixture(scope="module")
def random_vectors():
    # Generate and return random vectors
    # random_vectors = np.random.normal(size=(1000, 100))
    random_vectors = make_blobs(n_samples=1000, n_features=100, centers=2)
    return random_vectors

@pytest.fixture(scope="module")
def tsne_embedding_visualizer():
    tsne_embedding_visualizer = clevr_parser.EmbeddingVisualizer(backend='tsne').get_backend(identifier='tsne')
    return tsne_embedding_visualizer

def test_tsne_embedding_visualizer(tsne_embedding_visualizer, random_vectors):
    plt = tsne_embedding_visualizer.draw_embeddings(random_vectors[0])
    assert plt is not None

def test_tsne_cluster_visualizer(tsne_embedding_visualizer, random_vectors):
    # This function uses clustering to get the labels
    plt = tsne_embedding_visualizer.draw_embeddings(random_vectors[0], show_clusters=True, n_clusters=2)
    assert plt is not None

def test_tsne_true_cluster_visualizer(tsne_embedding_visualizer, random_vectors):
    # The function supplies the true cluster labels
    plt = tsne_embedding_visualizer.draw_embeddings(random_vectors[0], labels=random_vectors[1], show_clusters=True,
                                                    n_clusters=2)
    assert plt is not None

def test_neighbors(tsne_embedding_visualizer, random_vectors):
    n_neighbors = 3
    neighbors = tsne_embedding_visualizer.get_nearest_neighbors(random_vectors[0], n_neighbors=n_neighbors)
    print(neighbors)
    assert neighbors.shape == (random_vectors[0].shape[0], n_neighbors)

    pivots = [0, 3, 5]
    neighbors = tsne_embedding_visualizer.get_nearest_neighbors(random_vectors[0], n_neighbors=n_neighbors,
                                                                pivots=pivots)
    print(neighbors)
    assert len(neighbors) == len(pivots) and (len(neighbors[pivots[0]]) == n_neighbors)
