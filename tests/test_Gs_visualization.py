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


@pytest.fixture(scope="module")
def parser():
    parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                 has_spatial=True,
                                 has_matching=True).get_backend(identifier='spacy')
    return parser


def test_visualize_and_mat_spa(parser):
    # for dist in ['val']:
    for dist in ['train', 'val']:
        s_ams = get_s_sample("and_mat_spa", dist)
        print(f"s_ams_{dist} = {s_ams}")
        Gs, doc = parser.parse(s_ams, return_doc=True)
        ax_title = f"{doc}"
        Gs, en_graphs = parser.get_nx_graph_from_doc(doc)
        G = parser.draw_graph(Gs, en_graphs, ax_title=ax_title, doc=doc)
        assert G is not None


def test_visualize_or_mat_spa(parser):
    for dist in ['train', 'val']:
        s_oms = get_s_sample("or_mat_spa", dist)
        print(f"s_oms_{dist} = {s_oms}")
        Gs, doc = parser.parse(s_oms, return_doc=True)
        ax_title = f"{doc}"
        Gs, en_graphs = parser.get_nx_graph_from_doc(doc)
        G = parser.draw_graph(Gs, en_graphs, ax_title=ax_title, doc=doc)
        assert G is not None