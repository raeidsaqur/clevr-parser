#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: test_Gu_visualization.py
# Author: raeidsaqur
# Email: rsaqur@cs.princeton.edu
# Created on: 2020-05-03
# 
# This file is part of RSMLKit
# Distributed under terms of the MIT License

import os

import clevr_parser
import clevr_parser.utils as parser_utils
import pytest

from .samples import get_s_sample


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
    Gu, left_part, right_part = parser_utils.compose_multimodal_graphs(Gs, Gt)

    ax_title = f"{s_doc}"

    # Test graphviz
    G = gviz_visualizer.draw_graph(Gu, left_part, right_part,\
            save_file_path=os.path.join(create_output_dir, "and_mat_spa_"+split+".svg"),\
            ax_title=ax_title, format='svg', attr_node_label=True, show_edge_labels=True)
    assert G is not None