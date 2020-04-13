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

def demo_G_text_spatial_relation(parser, text=None):
    text = "The sphere is behind a rubber cylinder right of a metal cube" if text is None else text
    q_graph, q_doc = parser.parse(text, return_doc=True)
    #parser.visualize(q_doc)
    ax_title = f"{q_doc}"
    G_text, en_graphs = parser.get_nx_graph_from_doc(q_doc)
    G = parser.draw_graph(G_text, en_graphs, ax_title=ax_title, doc=q_doc)

def get_s_sample(template:str) -> str:
    # Question samples from [and|or_mat_spa] #
    """
    [and_mat_spa_baseline]
    Final program module = query_color
    Question type: query, answer: cyan, question for CLEVR_val_011582.png
    """
    s_ams_bline = "There is a thing that is on the right side of the tiny cyan rubber thing " \
                  "and to the left of the large green matte cylinder; what is its color?"

    """
    [and_mat_spa_val]
    Final program module = query_size
    Question type: query, answer: small, question for CLEVR_val_000019.png, 
    """
    s_ams_val = "What is the size of the thing that is in front of the big yellow object " \
                "and is the same shape as the big green thing?"

    """
    Final program module = count
    Question type: count, answer: 2, question for CLEVR_val_008452.png, 
    """
    s_oms_bline = "How many things are either small green objects in front of the small purple cylinder " \
                  "or large metallic things that are behind the red matte thing ?"

    """
    Final program module = count
    Question type: count, answer: 2, question for CLEVR_val_000439.png, 
    """
    s_oms_val = "How many things are cylinders that are behind the large purple metal thing " \
                "or purple cylinders that are the same size as the cyan thing ?"

    if template == "and_mat_spa_baseline":
        return s_ams_bline
    elif template == "and_mat_spa_val":
        return s_ams_val
    elif template == "or_mat_spa_baseline":
        return s_oms_bline
    elif template == "or_mat_spa_val":
        return s_oms_val
    else:
        raise ValueError("template must be one of [and|or]_mat_spa_[baseline|val]")
    return None


def main():
    clevr_img_name = lambda split, i: f"CLEVR_{split}_{i:06d}.png"
    #Load Parser
    parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm', has_spatial=True).get_backend(identifier='spacy')
    clevrr_baseline_qp = "../data/CLEVRR_v1.0/questions/CLEVRR_compare_baseline_questions.json"
    image_grounding_parsed_gp = "../data/CLEVR_v1.0/scenes_parsed/val_scenes_parsed.json"

    s_ams_bline = get_s_sample("and_mat_spa_baseline")
    demo_G_text_spatial_relation(parser, text=s_ams_bline)

    print("done")


if __name__ == '__main__':
    main()

