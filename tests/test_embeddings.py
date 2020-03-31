#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
np = pytest.importorskip('numpy')
nx = pytest.importorskip('networkx')
import os, sys, platform
import json

nb_dir = os.getcwd().split()[0]
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)

import clevr_parser
from clevr_parser.utils import *

import matplotlib.pyplot as plt

# Captions for 2obj_train_000001A
cap_2obj_1a = 'A small red rubber cylinder is behind a large brown metal sphere'
cap_2obj_1b = 'The large brown metal sphere has a small red rubber cylinder to the right'
cap_2obj_1c = 'The large brown metal sphere is in front of a small red rubber cylinder'

# ------ 1obj -------- #
# Captions for 1obj_train: CLEVR_train_000000.png
cap_1obj_train_00 = 'A large rubber purple sphere'
cap_1obj_train_00B = 'A rubber purple sphere'

@pytest.fixture(scope="module")
def parser():
    parser = clevr_parser.Parser().get_backend(identifier='spacy', model='en_core_web_sm')
    return parser


def test_parser_1obj_graph_embedding(parser):
    caption = "small red rubber ball"
    Gs, doc = parser.parse(caption, return_doc=True)
    dim = 96
    nodes = Gs.nodes(data=True)
    N = len(nodes)
    M = dim
    feat_mat = parser.get_embeddings(Gs, doc, embd_dim=dim)
    assert feat_mat is not None
    assert feat_mat.shape == (N, M)


def test_parser_2obj_graph_embedding(parser):
    caption = "There is a green metal block; the tiny metal thing is to the left of it"
    Gs, doc = parser.parse(caption, return_doc=True)
    dim = 96
    nodes = Gs.nodes(data=True)
    N = len(nodes)
    M = dim
    feat_mat = parser.get_embeddings(Gs, doc, embd_dim=dim)
    assert feat_mat is not None
    assert feat_mat.shape == (N, M)

def test_parser_1obj_embedding_ordering(parser):
    '''
    The embedding ordering should be: <obj>, <Z>, <C>, <M>, <S>
    '''
    caption = "small red rubber ball"   # <Z> <C> <M> <S>
    caption2 = "red rubber small ball"  # <C> <M> <Z> <S>
    Gs, doc = parser.parse(caption, return_doc=True)
    Gs2, doc2 = parser.parse(caption2, return_doc=True)
    dim = 96
    nodes = Gs.nodes(data=True)
    N = len(nodes); M = dim
    feat_mat = parser.get_embeddings(Gs, doc, embd_dim=dim)

    nodes2 = Gs2.nodes(data=True)
    N2 = len(nodes2)
    assert N == N2
    feat_mat2 = parser.get_embeddings(Gs2, doc2, embd_dim=dim)
    eps = 1e-4
    delta = np.subtract(feat_mat, feat_mat2)
    delta = delta.sum()

    assert feat_mat.shape == (N, M)
    assert feat_mat2.shape == (N2, M)


