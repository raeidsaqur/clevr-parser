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
                                 has_matching=False).get_backend(identifier='spacy')
    return parser


def test_visualize_and_mat_spa(parser):
    s_ams_bline = get_s_sample("and_mat_spa", "train")
    Gs, doc = parser.parse(s_ams_bline, return_doc=True)
    ax_title = f"{doc}"
    Gs, en_graphs = parser.get_nx_graph_from_doc(doc)
    G = parser.draw_graph(Gs, en_graphs, ax_title=ax_title, doc=doc)
    assert G is not None