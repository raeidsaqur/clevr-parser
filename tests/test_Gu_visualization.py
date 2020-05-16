#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: test_Gu_visualization.py
# Author: raeidsaqur
# Email: rsaqur@cs.princeton.edu
# Created on: 2020-05-03
# 
# This file is part of RSMLKit
# Distributed under terms of the MIT License

import pytest
import os, sys, platform
import json
import matplotlib.pyplot as plt

import clevr_parser
import clevr_parser.utils as parser_utils
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

@pytest.fixture(scope="module")
def create_output_dir():
    dir_name = 'tests_output'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name    

clevr_img_name = lambda split, i: f"CLEVR_{split}_{i:06d}.png"

def test_visualize_Gu(parser, plt_visualizer, gviz_visualizer, create_output_dir):
    """ Presumes image scene graphs are available in designated folder """
    ## Get Image Graph (Gt) ##
    img_idx = 19; split = 'val'
    img_fn = clevr_img_name(split, img_idx)
    fp = "../data/CLEVR_v1.0/scenes_parsed/val_scenes_parsed.json"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} does not exist")
    img_scene = parser_utils.load_grounding_for_img(img_fn, fp)
    try:
        Gt, t_doc = parser.get_doc_from_img_scene(img_scene)
    except AssertionError as ae:
        print(f"AssertionError Encountered: {ae}")

    ## Get a corresponding question Graph (Gs) ##
    #"What is the size of the thing that is in front of the big yellow object " \
    #             "and is the same shape as the big green thing?"
    s_ams_val = get_s_sample("and_mat_spa", dist=split)
    Gs, s_doc = parser.parse(s_ams_val, return_doc=True)

    # Get full graph composed of Gs, Gt #
    Gu, left_part, right_part = parser_utils.compose_multimodal_graphs(Gs, Gt, connect_obj_nodes=True)
    """
    RS Notes:
    print(Gu.nodes(data=True))
    Gs has 10 nodes, Gt has 20 nodes, Gu has (10 + 20) 30 nodes
    The visualization task is to draw this composed graph. N.b. the node labes are prefixed with 'Gs' 'Gt' 
    Also, maybe the left (source) and right (target) partitions can be helpful for setting up the layout?
    
    [('Gs-obj', {'label': 'CLEVR_OBJ', 'val': 'thing'}), ('Gs-<S>', {'label': 'shape', 'val': 'thing'}), 
    ('Gs-obj2', {'label': 'CLEVR_OBJ', 'val': 'big yellow object'}), ('Gs-<Z2>', {'label': 'size', 'val': 'big'}), 
    ('Gs-<C2>', {'label': 'color', 'val': 'yellow'}), ('Gs-<S2>', {'label': 'shape', 'val': 'object'}), 
    ...
    ('Gt-obj', {'label': 'CLEVR_OBJ', 'val': 'small brown metal cube', 'pos': (0.9492958188056946, 0.14152207970619202, 
    0.35215944051742554)}), ('Gt-<Z>', {'label': 'size', 'val': 'small'}), ('Gt-<C>', {'label': 'color', 'val': 'brown'}), 
    ('Gt-<M>', {'label': 'material', 'val': 'metal'}), ('Gt-<S>', {'label': 'shape', 'val': 'cube'}), 
    ('Gt-obj2',  
    ....
    ('Gt-<C4>', {'label': 'color', 'val': 'yellow'}), 
    ('Gt-<M4>', {'label': 'material', 'val': 'metal'}), 
    ('Gt-<S4>', {'label': 'shape', 'val': 'cube'})]
    """
    ax_title = f"{s_doc}"
    # Test graphviz
    G = gviz_visualizer.draw_graph(Gu,\
            save_file_path=os.path.join(create_output_dir, "and_mat_spa_"+split+".svg"), ax_title=ax_title)
    assert G is not None