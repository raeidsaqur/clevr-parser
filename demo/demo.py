#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : demo.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/22/2019
#
# This file is part of PGFM Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/pgfmParser

"""
A small demo for the scene graph parser.
"""
import os, sys, platform
import json

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)

print(f"{os.name}.{platform.system()}.{platform.release()}.{platform.node()}")

import clevr_parser
from clevr_parser.utils import *
from tests import TEMPLATES, get_s_sample

from typing import *

def demo_closure(parser):
    q = "Is there a blue thing that is the same size as the brown shiny object in front of the gray matte sphere?"
    _, doc = parser.parse(q)
    parser.visualize(doc, dep=False)


def demo_G_text(parser, text=None):
    text = "Is the gray matte object the same size as the green rubber cylinder" if text is None else text
    q_graph, q_doc = parser.parse(text, return_doc=True)
    ax_title = f"{q_doc}"
    G_text, en_graphs = parser.get_nx_graph_from_doc(q_doc)
    G = parser.draw_graph(G_text, en_graphs, ax_title=ax_title)


def demo_G_scene(parser, gfp):
    from clevr_parser import utils
    groundings = utils.load_groundings_from_path(gfp)
    g = groundings[0]
    G_img = parser.draw_clevr_img_scene_graph(g)

def demo_Gs_spatial_relation(parser, text=None):
    text = "The sphere is behind a rubber cylinder right of a metal cube" if text is None else text
    q_graph, q_doc = parser.parse(text, return_doc=True)
    #parser.visualize(q_doc)
    ax_title = f"{q_doc}"
    G_text, en_graphs = parser.get_nx_graph_from_doc(q_doc)
    G = parser.draw_graph(G_text, en_graphs, ax_title=ax_title, doc=q_doc)

def demo_visualize_and_mat_spa(parser):
    # for dist in ['val']:
    for dist in ['train', 'val']:
        s_ams = get_s_sample("and_mat_spa", dist)
        print(f"s_ams_{dist} = {s_ams}")
        Gs, doc = parser.parse(s_ams, return_doc=True)
        ax_title = f"{doc}"
        _, en_graphs = parser.get_nx_graph_from_doc(doc)
        G = parser.draw_graph(Gs, en_graphs, doc=doc, ax_title=ax_title )
        assert G is not None

def demo_visualize_or_mat_spa(parser):
    for dist in ['train', 'val']:
        s_oms = get_s_sample("or_mat_spa", dist)
        print(f"s_oms_{dist} = {s_oms}")
        Gs, doc = parser.parse(s_oms, return_doc=True)
        ax_title = f"{doc}"
        _, en_graphs = parser.get_nx_graph_from_doc(doc)
        G = parser.draw_graph(Gs, en_graphs, ax_title=ax_title, doc=doc)
        assert G is not None

def demo_visualize_Gt(parser, fp, img_fn=None, img_idx=None):
    """ Demo of a image scene graph generation """
    # fp = "../data/CLEVR_v1.0/scenes_parsed/train_scenes_parsed.json"
    fp = "../data/CLEVR_v1.0/scenes_parsed/val_scenes_parsed.json"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} does not exist")

    img_scene = None; scenes = None
    if img_idx:
        img_scene = load_grounding_for_img_idx(img_idx, fp)
    elif img_fn:
        img_scene = load_grounding_for_img(img_fn)
    else:
        scenes = load_groundings_from_path(fp)
    try:
        Gt, t_doc = parser.get_doc_from_img_scene(img_scene)
    except AssertionError as ae:
        print(f"AssertionError Encountered: {ae}")
        print(f"[{img_fn}] Excluding images with > 10 objects")
    ax_title = f"{t_doc}"
    Gt, en_graphs = parser.get_nx_graph_from_doc(t_doc)
    parser.draw_graph(Gt, en_graphs, ax_title=ax_title)

def main():
    clevr_img_name = lambda split, i: f"CLEVR_{split}_{i:06d}.png"
    #clevrr_baseline_qp = "../data/CLEVRR_v1.0/questions/CLEVRR_compare_baseline_questions.json"
    image_grounding_parsed_gp = "../data/CLEVR_v1.0/scenes_parsed/val_scenes_parsed.json"
    parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                 has_spatial=True,
                                 has_matching=True).get_backend(identifier='spacy')
    #s_ams_bline = get_s_sample(template="and_mat_spa", dist="train")
    # demo_Gs_spatial_relation(parser, text=s_ams_bline)
    # demo_visualize_and_mat_spa(parser)
    #demo_visualize_or_mat_spa(parser)
    # CLEVR_val_000015.png
    demo_visualize_Gt(parser, image_grounding_parsed_gp, img_idx=15)
    print("done")


if __name__ == '__main__':
    main()

