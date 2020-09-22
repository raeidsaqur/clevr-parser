#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
np = pytest.importorskip('numpy')
nx = pytest.importorskip('networkx')
import os, sys

nb_dir = os.getcwd().split()[0]
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)

import clevr_parser
from clevr_parser.utils import *

# ------ 1obj -------- #
# Captions for 1obj_train: CLEVR_train_000000.png
cap_1obj_train_00 = 'A large rubber purple sphere'
cap_1obj_train_00B = 'A rubber purple sphere'

# Captions for 2obj_train_000001A
cap_2obj_1a = 'A small red rubber cylinder is behind a large brown metal sphere'
cap_2obj_1b = 'The large brown metal sphere has a small red rubber cylinder to the right'
cap_2obj_1c = 'The large brown metal sphere is in front of a small red rubber cylinder'


@pytest.fixture(scope="module")
def parser():
    parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                 has_spatial=True,
                                 has_matching=True).get_backend(identifier='spacy')
    return parser

@pytest.fixture(scope="module")
def embedder(parser):
    embedder = clevr_parser.Embedder(backend='torch', parser=parser).get_backend(identifier='torch')
    return embedder

def test_parser_1obj_1attr_graph_embedding(parser, embedder):
    s = "thing"
    is_padding_pos = False
    Gs, doc = parser.parse(s, return_doc=True)
    dim = 96
    N, M = len(Gs.nodes()), (dim+3) if is_padding_pos else dim
    Xs, ei, e_attr = embedder.embed_s(s, is_padding_pos=is_padding_pos)
    assert Xs is not None
    assert Xs.shape == (N, M)
    if e_attr is not None:
        assert e_attr.shape == (len(Gs.edges()), M)

def test_parser_1obj_graph_embedding(parser, embedder):
    s = "small red rubber ball"
    is_padding_pos = True
    Gs, doc = parser.parse(s, return_doc=True)
    dim = 96
    N, M = len(Gs.nodes()), (dim+3) if is_padding_pos else dim

    Xs, ei, e_attr = embedder.embed_s(s)
    assert Xs is not None
    assert Xs.shape == (N, M)
    if e_attr is not None:
        assert e_attr.shape == (len(Gs.edges()), M)

def test_parser_2obj_graph_embedding(parser, embedder):
    s = "There is a green metal block; the tiny metal thing is to the left of it"
    Gs, doc = parser.parse(s, return_doc=True)
    dim = 96
    N = len(Gs.nodes())
    for is_padding_pos in [True, False]:
        M = (dim+3) if is_padding_pos else dim
        Xs, ei, e_attr = embedder.embed_s(s, embd_dim=dim, is_padding_pos=is_padding_pos)
        assert Xs is not None
        assert Xs.shape == (N, M)
        if e_attr is not None:
            assert e_attr.shape == (len(Gs.edges()), M)

def test_parser_1obj_embedding_ordering(parser, embedder):
    '''
    Permutation Equivariance Test
    The embedding ordering should be: <obj>, <Z>, <C>, <M>, <S>
    '''
    s = "small red rubber ball"   # <Z> <C> <M> <S>
    s2 = "red rubber small ball"  # <C> <M> <Z> <S>

    Gs, doc = parser.parse(s, return_doc=True)
    Gs2, doc2 = parser.parse(s2, return_doc=True)
    dim = 96
    nodes = Gs.nodes(data=True)

    for is_padding_pos in [True, False]:
        N, M = len(nodes), (dim+3) if is_padding_pos else dim
        Xs, ei, e_attr = embedder.embed_s(s, embd_dim=dim, is_padding_pos=is_padding_pos)
        Xs2, ei2, e_attr2 = embedder.embed_s(s2, embd_dim=dim, is_padding_pos=is_padding_pos)

        N2 = len(Gs2.nodes())
        assert N == N2
        # eps = 1e-4
        # delta = np.subtract(Xs, Xs2)
        # delta = delta.sum()
        assert Xs.shape == (N, M)
        assert Xs2.shape == (N2, M)


