#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/21/2019
#
# This file is part of PGFM Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

import functools
import tabulate
from typing import List, Dict, Any, Collection, Set
import os,json
import spacy
from spacy import displacy
from .explacy import *

from matplotlib import pyplot, patches


__all__ = ['combine_en_graphs','compose_multimodal_graphs', 'draw_adjacency_matrix',
           'trace', 'print_eq', 'print_star', 'print_dash', 'tprint',
           'print_parsed_doc', 'print_parsed_caption', 'visualize_parsed',
           'dotdict']


print_dash = lambda x: print(f"\n" + "-" * x + "\n")
print_star = lambda x: print(f"\n" + "*" * x + "\n")
print_eq = lambda x: print(f"\n" + "=" * x   + "\n")

### Networkx Library Extensions and Helpers ####
import networkx as nx
from networkx.algorithms import bipartite
from networkx.linalg import adjacency_matrix, adj_matrix, laplacian_matrix, normalized_laplacian_matrix

def combine_en_graphs(en_s:Dict, en_t:Dict) -> Dict:
    """ Remove: temporary helper for aiding with drawing"""
    en_graphs_union = {}
    count = 1
    for d in (en_s, en_t):
        for k, v in d.items():
            en_graphs_union[count] = v
            count += 1
    return en_graphs_union

def compose_multimodal_graphs(Gs: nx.Graph, Gt: nx.Graph,
                              connect_obj_nodes=False, obj_node_id='obj'):

    """
    Compose a combined graph from src and target mutlimodal graph representations
    :param Gs: The source Graph, i.e., the text graph representation
    :param Gt: The target Graph, i.e., the grounding (image features) graph representation
    :param connect_obj_nodes: flag to set whether the obj nodes between graphs should be connected
    :param obj_node_id: the identifier for determining a obj node
    :return: G_union, and the two bi-partite graph node sets.
    """
    G_union = nx.union(Gs, Gt, rename=('Gs-', 'Gt-'))  # Rename to avoid node name collisions with prefix
    left_part = list(filter(lambda n: 'Gs-' in n, G_union.nodes()))
    left_part, right_part = nx.bipartite.sets(G_union, left_part)  # ensure valid bipartite graph

    ### Add edges between all Gs -> Gt obj nodes. We need to learn these edge weights ###
    if connect_obj_nodes:
        is_head_node = lambda x: obj_node_id in x
        Gs_head_nodes = sorted(list(filter(is_head_node, left_part)))
        Gt_head_nodes = sorted(list(filter(is_head_node, right_part)))
        for i in Gs_head_nodes:
            for j in Gt_head_nodes:
                G_union.add_edge(i, j)

    return G_union, left_part, right_part



def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    # Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5))  # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                           len(module),  # Width
                                           len(module),  # Height
                                           facecolor="none",
                                           edgecolor=color,
                                           linewidth="1"))
            current_idx += len(module)


### END: Networkx Library Extensions and Helpers ####

### Decorators ###

def trace(f, DEBUG=False):
    def wrap(*args, **kwargs):
        if DEBUG:
            print_dash(50)
            print(f"[TRACE] func: {f.__name__}, args: {args}, kwargs: {kwargs}")
            print_dash(50)
        return f(*args, **kwargs)
    return wrap

### End Decorators ###

#TODO: import graph library here
@trace
def toGraph(graph:Dict, id:int=0, file:str=None, caption:str=None, out=None):
    # Base graph lib https://github.com/networkx/networkx
    raise NotImplementedError

@trace
def toJSON(graph:Dict, id:int, file:str=None, caption:str=None, out=None):
    """
    Form and output a json object from the give graph.
    if out is not None, then create an out file, o.w. stdout.
    :param graph:
    :param id:
    :param file: the corresponding image filename
    :param caption:
    :param out: JSON output file
    :return:
    """
    _objects = []
    _relations = []
    json_template = {
        "image_index": id,
        "image_filename": file,
        "objects": _objects,
    }
    raise NotImplementedError

@trace
# def print_parsed_caption(graph:Dict, nlp, id:int=0, file:str=None, caption:str=None, out=None):
def print_parsed_caption(caption:str, nlp=None, visualize=False):
    assert caption is not None
    if nlp is None:
        nlp = spacy.load('en') # load default spacy, N.b. won't contain Clevr entity recognizer

    doc = nlp(caption)
    print_parsed_doc(doc, visualize)

def print_parsed_doc(doc, visualize=False):
    print_parse(doc)
    if visualize:
        visualize_parsed(doc, dep=True)


def visualize_parsed(doc, dep=False):
    displacy.render(doc, style='ent', jupyter=True)
    if dep:
        displacy.render(doc, style='dep', jupyter=True, options={'distance': 70})


@trace
def tprint(graph, file=None, show_entities=True, show_relations=True):
    """
    Print a scene graph as a table.
    The printed strings contains only essential information about the parsed scene graph.
    """

    _print = functools.partial(print, file=file)

    if show_entities:
        _print('Entities:')

        entities_data = [
            [e['head'].lower(), e['span'].lower(), ','.join([ x['span'].lower() for x in e['modifiers'] ])]
            for e in graph['entities']
        ]
        _print(tabulate.tabulate(entities_data, headers=['Head', 'Span', 'Modifiers'], tablefmt=_tabulate_format))

    if show_relations:
        _print('Relations:')

        entities = graph['entities']
        relations_data = [
            [
                entities[rel['subject']]['head'].lower(),
                rel['relation'].lower(),
                entities[rel['object']]['head'].lower()
            ]
            for rel in graph['relations']
        ]
        _print(tabulate.tabulate(relations_data, headers=['Subject', 'Relation', 'Object'], tablefmt=_tabulate_format))


_tabulate_format = tabulate.TableFormat(
        lineabove=tabulate.Line("+", "-", "+", "+"),
        linebelowheader=tabulate.Line("|", "-", "+", "|"),
        linebetweenrows=None,
        linebelow=tabulate.Line("+", "-", "+", "+"),
        headerrow=tabulate.DataRow("|", "|", "|"),
        datarow=tabulate.DataRow("|", "|", "|"),
        padding=1, with_header_hide=None
)


def load_questions_and_groundings(qfp, gfp) -> (List[Dict], List[Dict]):
    if not os.path.exists(qfp):
        raise FileNotFoundError(f"{qfp} does not exist")
    if not os.path.exists(gfp):
        raise FileNotFoundError(f"{gfp} does not exist")
    questions = load_questions(qfp)
    groundings = load_groundings_for_questions(questions, gfp)

    return (questions, groundings)

def load_questions(fp) -> List[Dict]:
    """
    :param fp: File Path to the questions/captions file
    :return: Dict containing a list of question objects
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} does not exist")
    questions = None
    with open(fp, 'r') as f:
        questions = json.load(f)["questions"]
    assert len(questions) > 1
    return questions


def load_grounding_for_questionObj(qObj:Dict, fp) -> Dict:
    scenes = load_groundings_from_path(fp)
    assert qObj['image_filename'] is not None
    scene = list(filter(lambda g: g['image_filename'] == qObj['image_filename']))
    return scene[0]

def load_grounding_for_img(fn:str, fp):
    groundings = load_groundings_from_path(fp)
    grounding = list(filter(lambda x: x['image_filename'] == fn, groundings))[0]
    return grounding

def load_groundings_from_path(fp) -> List[Dict]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} does not exist")
    scenes = None
    with open(fp, 'r') as f:
        scenes = json.load(f)["scenes"]
    assert len(scenes) > 1
    return scenes

def load_groundings_for_questions(tobjs:List[Dict], fp) -> List[Dict]:
    """
    N.b. the groundings path could refer to the full scenes graph path, or
    the parsed scene graph (derived from the image segmentation pipeline.

    :param tobjs: A list of question/caption objects
    :param fp: file path to the groundings.
    :return: A list of grounding objects corresponding to the text
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} does not exist")

    scenes = load_groundings_from_path(fp)
    tscenes = []
    for t in tobjs:
        scene = list(filter(lambda g: g['image_filename'] == t['image_filename'], scenes))
        tscenes.append(scene[0])

    assert len(tobjs) == len(tscenes)
    return tscenes


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__