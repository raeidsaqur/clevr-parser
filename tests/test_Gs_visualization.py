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

# Test Fixtures #
@pytest.fixture(scope="module")
def parser():
    parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                 has_spatial=True,
                                 has_matching=True).get_backend(identifier='spacy')
    return parser

@pytest.fixture(scope="module")
def plt_visualizer():
    plt_visualizer = clevr_parser.Visualizer(backend='matplotlib').get_backend(identifier='matplotlib')
    return plt_visualizer

@pytest.fixture(scope="module")
def gviz_visualizer():
    gviz_visualizer = clevr_parser.Visualizer(backend='graphviz').get_backend(identifier='graphviz')
    return gviz_visualizer

def test_visualize_and_mat_spa(parser, plt_visualizer, gviz_visualizer):
    # for dist in ['val']:
    for dist in ['train', 'val']:
        s_ams = get_s_sample("and_mat_spa", dist)
        print(f"s_ams_{dist} = {s_ams}")
        Gs, doc = parser.parse(s_ams, return_doc=True)
        ax_title = f"{doc}"
        _, en_graphs = parser.get_nx_graph_from_doc(doc)
        # G = parser.draw_graph_testing(Gs, en_graphs, doc=doc, ax_title=ax_title )
        G = plt_visualizer.draw_graph(Gs, doc=doc, ax_title=ax_title)
        assert G is not None

def test_visualize_or_mat_spa(parser, plt_visualizer, gviz_visualizer):
    for dist in ['train', 'val']:
        s_oms = get_s_sample("or_mat_spa", dist)
        print(f"s_oms_{dist} = {s_oms}")
        Gs, doc = parser.parse(s_oms, return_doc=True)
        ax_title = f"{doc}"
        _, en_graphs = parser.get_nx_graph_from_doc(doc)
        #G = parser.draw_graph_testing(Gs, en_graphs, ax_title=ax_title, doc=doc)
        G = plt_visualizer.draw_graph(Gs, doc=doc, ax_title=ax_title)
        assert G is not None