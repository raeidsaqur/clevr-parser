#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: test_Gt_visualization.py
# Author: Raeid Saqur
# Email: rsaqur@cs.princeton.edu
# Created on: 2020-05-03
# 
# This file is part of CLEVR Parser
# Distributed under terms of the MIT License

import pytest
import os, sys, platform
import json
import matplotlib.pyplot as plt

import clevr_parser
import clevr_parser.utils as parser_utils

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

clevr_img_name = lambda split, i: f"CLEVR_{split}_{i:06d}.png"

def test_visualize_Gt(parser, plt_visualizer, gviz_visualizer):
    """ Presumes image scene graphs are available in designated folder """
    img_idx = 15; split = 'val'
    img_fn = clevr_img_name(split, img_idx)
    fp = "../data/CLEVR_v1.0/scenes_parsed/val_scenes_parsed.json"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} does not exist")
    img_scene = parser_utils.load_grounding_for_img(img_fn, fp)
    try:
        Gt, t_doc = parser.get_doc_from_img_scene(img_scene)
    except AssertionError as ae:
        print(f"AssertionError Encountered: {ae}")
    ax_title = f"{t_doc}"
    G = plt_visualizer.draw_graph(Gt, doc=t_doc, ax_title=ax_title)
    assert G is not None