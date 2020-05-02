#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: visualizer.py
# Author:
# Email:
# Created on: 2020-05-02
#
# This file is part of CLEVR-PARSER
# Distributed under terms of the MIT License
# https://github.com/raeidsaqur/clevr-parser

from ..visualizer import Visualizer, get_default_visualizer
from ..parser import  Parser, get_default_parser
from .backend import VisualizerBackend
from ..utils import *
from typing import List, Dict, Tuple, Sequence
import logging
logger = logging.getLogger(__name__)
import os, copy

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import pygraphviz as pgv
    import networkx as nx
except ImportError as ie:
    logger.error(f"Install NetworkX: {ie.name}")

__all__ = ['GraphvizVisualizer']

@Visualizer.register_backend
class GraphvizVisualizer(VisualizerBackend):
    """
    Visualizer for CLEVR Graphs
    """
    __identifier__ = 'graphviz'

    def __init__(self):
        super().__init__()

    @classmethod
    def draw_graph(cls, G: nx.Graph, *args, **kwargs):
        """
        :param G: The input Graph of type nx.MultiGraph (undirectional)
                  or nx.MultiDiGraph (Directional)
        :param args: Additional args
        :param kwargs: Additoinal kw arguments like pos (for image_scene_graphs), ax_title etc.
        :return:
        """
        return cls.draw_graph_graphviz(G, *args, **kwargs)

    @classmethod
    def draw_graph_graphviz(cls, G, pos=None, plot_box=False, ax_title=None):
        import random
        from networkx.drawing.nx_agraph import graphviz_layout

        NDV = G.nodes(data=True)
        NV = G.nodes(data=False)
        print(NDV)
        print(NV)
        EV = G.edges(data=False)
        EDV = G.edges(data=True)

        is_head_node = lambda x: 'obj' in x
        is_snode = lambda x: 'Gs' in x
        is_tnode = lambda x: 'Gt' in x

        # Desiderata:
        # Draw the head_nodes a little larger, node_size=60 for hnodes, and 40 for anodes
        # Color the Gs, Gt nodes differently or shape (node_shape)

        # nsz = [60 if is_head_node(node) else 40 for node in NV]
        # ncol = ['tab:purple' if is_snode(node) else 'tab:blue' for node in NV]
        # nshape = ['8' if is_head_node(node) else 'o' for node in NV]

        plt.figure(1, figsize=(8, 8))
        plt.axis('on' if plot_box == True else "off")
        plt.title(ax_title)
        if pos is None:
            pos = graphviz_layout(G, prog='neato')

        pos_shadow = copy.deepcopy(pos)
        shift_amount = 0.001
        for k, v in pos_shadow.items():
            x = v[0] + shift_amount
            y = v[1] - shift_amount
            pos_shadow[k] = (x, y)
            # pos_shadow[idx][0] += shift_amount
            # pos_shadow[idx][1] -= shift_amount

        # C = (G.subgraph(c) for c in nx.connected_components(G))
        # for g in C:
        #     c = [random.random()] * nx.number_of_nodes(g)  # random color..
        #     nx.draw(g, pos, node_size=40, node_color=c, vmin=0.0, vmax=1.0, with_labels=False)

        for n in NV:
            g = G.subgraph(n)
            nsz = 1200 if is_head_node(n) else 700
            # ncol = 'tab:purple' if is_snode(n) else 'tab:blue'
            # ref: https://matplotlib.org/examples/color/named_colors.html
            # ncol = 'b' if is_snode(n) else 'darkmagenta'
            ncol = 'b' if is_snode(n) else 'teal'
            # marker ref: https://matplotlib.org/api/markers_api.html#module-matplotlib.markers
            nshape = 'D' if is_head_node(n) else 'o'
            nx.draw(g, pos, node_size=nsz, node_color=ncol, node_shape=nshape, with_labels=True)
            nx.draw(g, pos_shadow, node_size=nsz, node_color='k', node_shape=nshape, alpha=0.2)

        nx.draw_networkx_edges(G, pos, edgelist=EDV)
        # nx.draw(G, pos, node_size=nsz, node_color=ncol, node_shape=nshape, vmin=0.0, vmax=1.0, with_labels=False)
        plt.show()

        return G