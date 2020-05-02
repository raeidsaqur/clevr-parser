#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
np = pytest.importorskip('numpy')
import os, sys, platform
import json
import matplotlib.pyplot as plt

import clevr_parser
from clevr_parser.utils import *
from  .samples import TEMPLATES, get_s_sample

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
    parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                 has_spatial=True,
                                 has_matching=False).get_backend(identifier='spacy')
    return parser

def test_parser_2obj(parser):
    caption = "The large brown metal sphere has a small red rubber cylinder to the right'"
    _, doc = parser.parse(caption, return_doc=True)
    num_clevr_objs = len(parser.filter_clevr_objs(doc.ents))
    assert num_clevr_objs == 2

def test_parser_1obj_entity_vector(parser):
    caption = "small red rubber ball"
    _, doc = parser.parse(caption, return_doc=True)
    ent_emd_sz = 384
    tok_emb_sz = int(ent_emd_sz / 4)
    entity = doc.ents[0]
    ent_vector = parser.get_clevr_entity_vector_embedding(entity, size=384)
    assert ent_vector is not None

    for i, token in enumerate(entity):
        idx = i*tok_emb_sz
        assert np.all(ent_vector[:, idx: (idx+tok_emb_sz)] == token.vector)


def test_parser_1obj_doc_vector(parser):
    caption = "small red rubber ball"
    _, doc = parser.parse(caption, return_doc=True)
    ent_emd_sz = 384
    doc_vector = parser.get_clevr_doc_vector_embedding(doc, ent_vec_size=ent_emd_sz)
    assert doc_vector is not None
    assert doc_vector.size == (ent_emd_sz * 1)


def test_parser_2obj_doc_vector(parser):
    caption = "There is a green metal block; the tiny metal thing is to the left of it"
    _, doc = parser.parse(caption, return_doc=True)
    ent_emd_sz = 384
    doc_vector = parser.get_clevr_doc_vector_embedding(doc, ent_vec_size=ent_emd_sz)
    assert doc_vector is not None
    assert doc_vector.size == (ent_emd_sz * 2)
    