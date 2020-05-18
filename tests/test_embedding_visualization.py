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
import os, sys, platform
import json
import matplotlib.pyplot as plt
from itertools import product

import clevr_parser

@pytest.fixture(scope="module")
def random_vectors():
    # Generate and return random vectors
    random_vectors = np.random.normal(size=(1000, 100))
    return random_vectors

@pytest.fixture(scope="module")
def tsne_embedding_visualizer():
    tsne_embedding_visualizer = clevr_parser.EmbeddingVisualizer(backend='tsne').get_backend(identifier='tsne')
    return tsne_embedding_visualizer

def test_tsne_embedding_visualizer(tsne_embedding_visualizer, random_vectors):
    plt = tsne_embedding_visualizer.draw_embeddings(random_vectors)
    assert plt is not None    