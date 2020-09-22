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

import logging

from .backend import VisualizerBackend
from .. import Embedder
from .. import Parser
from ..visualizer import Visualizer

logger = logging.getLogger(__name__)
import networkx as nx

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import pygraphviz as pgv
except ImportError as ie:
    logger.error(f"Install NetworkX: {ie.name}")

__all__ = ['GraphvizVisualizer']

@Visualizer.register_backend
class GraphvizVisualizer(VisualizerBackend):
    """
    Visualizer for CLEVR Graphs
    """
    __identifier__ = 'graphviz'

    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def draw_graph(cls, G: nx.Graph, *args, **kwargs):
        """
        :param G: The input Graph of type nx.MultiGraph (undirectional)
                  or nx.MultiDiGraph (Directional)
        :param args: Additional args
        :param kwargs: Additional kw arguments like pos (for image_scene_graphs), ax_title etc.
        :return:
        """
        # Detect if Gu is being drawn or Gs/Gt
        if all([('Gs' in node or 'Gt' in node) for node in tuple(G.nodes())]):
            return cls.draw_graph_graphviz_Gu(G, *args, **kwargs)
        else:
            return cls.draw_graph_graphviz(G, *args, **kwargs)

    @classmethod
    def draw_graph_graphviz(cls, G, save_file_path, en_graphs=None,
                            pos=None,
                            plot_box=False, ax_title=None,
                            head_node_label=True, attr_node_label=False,
                            show_edge_labels=False,
                            hnode_sz=0.5, anode_sz=0.5,
                            format='svg', dpi='100'):

        # Get the nodes and edges
        NDV = G.nodes(data=True)
        EDV = G.edges(data=True)

        # Classify into head nodes and attribute nodes
        _is_head_node = lambda x: 'obj' in x
        _is_attr_node = lambda x: 'obj' not in x

        # Instantiate a graph and set a high dpi
        A = pgv.AGraph(dpi = dpi, label=ax_title, labelloc='top')
        
        # Add nodes
        for node in NDV:
            # Get the graphviz attributes for the node
            attributes = cls.get_graphviz_attribute(node, EDV, anode_sz)

            # Add the node
            if _is_head_node(node[0]):
                if head_node_label:
                    A.add_node(node[0], width=hnode_sz, height=hnode_sz, fixedsize=True, **attributes)
                else:
                    A.add_node(node[0], width=hnode_sz, height=hnode_sz, fixedsize=True, **attributes, label='')
            else:
                if attr_node_label:
                    A.add_node(node[0], fixedsize=True, **attributes)
                else:
                    A.add_node(node[0], fixedsize=True, **attributes, label='')
        
        # Add edges
        for edge in EDV:
            if show_edge_labels:
                # If edge lable is empty, don't print it
                if not edge[2]:
                    A.add_edge(edge[0], edge[1])
                else:
                    if next(iter(edge[2])) in ['matching_re', 'spatial_re']:
                        A.add_edge(edge[0], edge[1], label=edge[2][next(iter(edge[2]))])
                    else:
                        A.add_edge(edge[0], edge[1], label=next(iter(edge[2])))
            else:
                A.add_edge(edge[0], edge[1])
        
        # Save the image
        A.draw(path=save_file_path, format=format, prog='neato')       

        return G

    @classmethod
    def draw_graph_graphviz_Gu(cls, G, ls, rs, save_file_path, en_graphs=None,
                            pos=None,
                            plot_box=False, ax_title=None,
                            head_node_label=True, attr_node_label=False,
                            show_edge_labels=False,
                            hnode_sz=0.5, anode_sz=0.5,
                            format='svg', dpi='100'):

        # Connect the corresponding nodes on the source and target side
        graph_parser = Parser(backend="spacy", model='en_core_web_sm',
                                           has_spatial=True,
                                           has_matching=True).get_backend(identifier='spacy')
        embedder = Embedder(backend='torch', parser=graph_parser).get_backend(identifier='torch')
        G = embedder.connect_matching_pair_edges(G, ls, rs)
        
        # Get the nodes and edges
        NDV = G.nodes(data=True)
        EDV = G.edges(data=True)

        # Classify into head nodes and attribute nodes
        _is_head_node = lambda x: 'obj' in x
        _is_attr_node = lambda x: 'obj' not in x

        # Classify into source nodes and target nodes
        _is_source_node = lambda x: 'Gs' in x
        _is_target_node = lambda x: 'Gt' in x      

        # Instantiate a graph and set a high dpi
        A = pgv.AGraph(dpi = dpi, label=ax_title, labelloc='top')
        
        # Add nodes
        for node in NDV:
            # Get the graphviz attributes for the node
            if _is_source_node(node[0]):
                attributes = cls.get_graphviz_attribute(node, EDV, anode_sz, isGs=True)
            else:
                attributes = cls.get_graphviz_attribute(node, EDV, anode_sz)
            
            # Add the node
            if _is_head_node(node[0]):
                if head_node_label:
                    A.add_node(node[0], width=hnode_sz, height=hnode_sz, fixedsize=True, **attributes, label=node[0].split('-')[-1])
                else:
                    A.add_node(node[0], width=hnode_sz, height=hnode_sz, fixedsize=True, **attributes, label='')
            else:
                # Draw only target side attribute nodes
                if _is_attr_node(node[0]) and _is_target_node(node[0]):
                    if attr_node_label:
                        A.add_node(node[0], fixedsize=True, **attributes, label=node[0].split('-')[-1])
                    else:
                        A.add_node(node[0], fixedsize=True, **attributes, label='')
        
        # Add edges
        for edge in EDV:
            # Don't draw source side node-attribute edges
            if (_is_attr_node(edge[0]) and _is_source_node(edge[1])) or (_is_source_node(edge[0]) and _is_attr_node(edge[1])):
                continue
            if show_edge_labels:
                # If edge lable is empty, don't print it
                if not edge[2]:
                    A.add_edge(edge[0], edge[1])
                else:
                    if next(iter(edge[2])) in ['matching_re', 'spatial_re']:
                        A.add_edge(edge[0], edge[1], label=edge[2][next(iter(edge[2]))])
                    else:
                        A.add_edge(edge[0], edge[1], label=next(iter(edge[2])))
            else:
                A.add_edge(edge[0], edge[1])
        
        # Save the image
        A.draw(path=save_file_path, format=format, prog='neato')       

        return G

    @classmethod
    def get_graphviz_attribute(cls, node, EDV=None, anode_sz=None, isGs=False):
        '''
        Returns the corresponding graphviz attribute to use

        Arguments:
            node:
            EDV:
            anode_sz:
            isGs: If it is a node from Gs when Gu is being drawn

        Returns:
            (shape, fillcolor, style, size)
        '''

        # Default node attributes
        default_shape = 'diamond'
        default_color = 'yellow'
        default_style = 'filled'

        # Head node attributes
        if isGs:
            head_shape = 'doublecircle'
            head_color = 'coral'
            head_style = 'filled'
        else:
            head_shape = 'doublecircle'
            head_color = 'aquamarine'
            head_style = 'filled'

        def get_color():
            '''
            Return the color of node to which the attribute is associated
            '''
            # Get the head node
            head_node = [u for u,v,d in EDV
                        if v == node[0] and 'obj' in u]
            assert len(head_node) == 1, "Attribute attached to two head nodes"
            head_node = head_node[0]

            # Get the associated color if it exists
            color = [d['color'] for u,v,d in EDV
                        if u == head_node and 'color' in d]
            if color:
                return color[0]
            else:
                return default_color

        
        shape_attr = {
            # Cylinder
            'cylinder': 'cylinder',
            'cylinders': 'cylinder',
            # Cube
            'cube': 'square',
            'cubes': 'square',
            'block': 'square',
            'blocks': 'square',
            # Sphere
            'sphere': 'circle',
            'spheres': 'circle',
            'ball': 'circle',
            'balls': 'circle',
            # Thing
            'thing': 'hexagon',
            'things': 'hexagon',
            'object': 'hexagon',
            'objects': 'hexagon',
            'default': 'circle'
        }

        material_attr = {
            'metal': ':white',
            'metallic': ':white',
            'shiny': ':white',
            'rubber': '',
            'matte': '',
            'default': ''
        }

        size_attr = {
            'large': 1.2,
            'big': 1.2,
            'small': 0.5,
            'tiny': 0.5,
            'default': 0.5
        }

        if 'CLEVR_OBJ' in node[1]['label']:
            return {'shape':head_shape, 'fillcolor':head_color, 'style':head_style}
        elif node[1]['label'] == 'shape':
            return {'shape':shape_attr.get(node[1]['val'], shape_attr['default']), 'fillcolor':get_color(), 'style':default_style, 'width':anode_sz, 'height':anode_sz}
        elif node[1]['label'] == 'color':
            return {'shape':default_shape, 'fillcolor':node[1]['val'], 'style':default_style, 'width':anode_sz, 'height':anode_sz}
        elif node[1]['label'] == 'material':
            return {'shape':default_shape, 'fillcolor':get_color()+material_attr.get(node[1]['val'], material_attr['default']), 'style':default_style, 'width':anode_sz, 'height':anode_sz}
        elif node[1]['label'] == 'size':
            return {'shape':default_shape, 'fillcolor':get_color(), 'style':default_style, 'width':size_attr.get(node[1]['val'], size_attr['default']), 'height':size_attr.get(node[1]['val'], size_attr['default'])}
        else:
            return {'shape':default_shape, 'fillcolor':get_color(), 'style':default_style}