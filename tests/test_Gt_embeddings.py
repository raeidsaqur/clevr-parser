#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
np = pytest.importorskip('numpy')
nx = pytest.importorskip('networkx')
import os, sys, platform
from itertools import product

nb_dir = os.getcwd().split()[0]
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)

import clevr_parser
from clevr_parser.utils import *


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

clevr_img_name = lambda split, i: f"CLEVR_{split}_{i:06d}.png"

def test_Gt_embedding(parser, embedder):
    img_idx = 15
    split = 'val'
    #img_fn = clevr_img_name(split, img_idx)
    fp = "../data/CLEVR_v1.0/scenes_parsed/val_scenes_parsed.json"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} does not exist")
    is_padding_pos = True
    dim = 96
    for is_padding_pos in [True, False]:
        Xt, ei, e_attr = embedder.embed_t(img_idx, fp, embd_dim=dim, is_padding_pos=is_padding_pos)
        M = (dim+3) if is_padding_pos else dim
        assert Xt is not None
        assert Xt.shape[1] == M
        if e_attr is not None:
            assert e_attr.shape[1] == M





