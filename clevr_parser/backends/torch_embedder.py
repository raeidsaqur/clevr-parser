#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : spacy_parser.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/21/2019
#
# This file is part of CLEVR Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

from .. import database
from ..embedder import Embedder
from ..parser import  Parser, get_default_parser
from .backend import EmbedderBackend, ParserBackend
from ..utils import *
from .spacy_parser import SpacyParser

from functools import reduce
from operator import itemgetter
import collections
from typing import List, Dict, Tuple, Sequence
import copy
import numpy as np
import scipy.sparse as sp

import logging
logger = logging.getLogger(__name__)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import pygraphviz as pgv
    import networkx as nx
    import torch
    import re
    import torch_geometric
    from torch_geometric.data import Data
except ImportError as ie:
    logger.error(f"Some required modules couldn't be imported: {ie.name}")

__all__ = ['TorchEmbedder', 'PairData']

class PairData(Data):
    def __inc__(self, key, value):
        if bool(re.search("^edge_[\w]_s$", key)):
            return self.x_s.size(0)
        if bool(re.search("^edge_[\w]*_t$", key)):
            return self.x_t.size(0)
        else:
            return 0

@Embedder.register_backend
class TorchEmbedder(EmbedderBackend):
    """
    Embeddings for CLEVR Graphs
    """
    __identifier__ = 'torch'

    def __init__(self, parser, model='en'):
        super().__init__()
        _parser = parser
        if not _parser:
            _parser = SpacyParser(model=model)
        self.__clevr_parser = _parser

    @property
    def clevr_parser(self):
        return self.__clevr_parser

    @clevr_parser.setter
    def clevr_parser(self, cp):
        self.__clevr_parser = cp

    ## Functions for Graph Embeddings ##
    def connect_matching_pair_edges(self, Gu: nx.MultiGraph,
                                    ls, rs, obj_node_id='obj',
                                    connect_obj_rel_edges=False) -> nx.Graph:
        """
        Ground truth (_gt_)generator function for a combined graph. Used for training S_0

        :param Gu: A (unconnected) composed graph of Gs, Gt. No relational links between head nodes,
        thus, the number_connected_components(Gs|Gt) = number of object nodes in each graph.


        :param Gs: The source Graph, i.e., the text graph representation
        :param Gt: The target Graph, i.e., the grounding (image features) graph representation
        :param obj_node_id: the identifier determining a obj (or head) node
        :return: matching_pairs: List of matching pair tuples between Gs, Gt
        """
        # matching_pairs: List[Tuple] = self.get_matching_pairs_in_bipartite_graph(Gu, ls, rs, obj_node_id)
        # Bug FIX: Changed matching_pairs = [('Gs-obj', None), ('Gs-obj2', 'Gt-obj')]
        # To: matching_pairs = [('Gs-obj2', 'Gt-obj')], unmatched_pairs = ['Gs-obj']
        matching_pairs, unmatched_pairs = self.get_matching_pairs_in_bipartite_graph(Gu, ls, rs, obj_node_id)
        NDV = Gu.nodes(data=True)
        # Connect the matching pairs if not connected #
        for pair in matching_pairs:
            s_node, t_node = pair
            if not Gu.has_edge(s_node, t_node):
                Gu.add_edge(s_node, t_node, '<gt>')
            # Connect Attr Nodes #
            is_head_node = lambda x: obj_node_id in x
            Ns = nx.neighbors(Gu, s_node)
            Nt = list(nx.neighbors(Gu, t_node))
            for ns in Ns:
                if is_head_node(ns):
                    continue
                # Check label equality only, 'val' equality already verified
                ns_label = NDV[ns]['label']
                logger.debug(f'Source attr node label = {ns_label}')
                # TODO: potential issue here, Nt should always have a matching attr node
                for nt in filter(lambda x: NDV[x]['label'] == ns_label, Nt):
                    if not Gu.has_edge(ns, nt):
                        Gu.add_edge(ns, nt, key='<gt>')
        if connect_obj_rel_edges:
            # TODO: Add a obj relation edge among all Gs, Gt obj nodes. Actually
            # should be done earlier in the parse cycle.
            pass
        return Gu

    def get_matching_pairs_in_bipartite_graph(self, Gu: nx.MultiGraph, ls, rs, obj_node_id='obj') -> (
    List[Tuple], List):
        """
        Compares source and target nodes by labels and values and returns the matching pairs, or unmatched pairs
        if there are source nodes that can't be matched.
        N.b. the presence of a single unmatched Gs node suffices to deem the Gu as unmatched
        :param Gu:
        :param ls:
        :param rs:
        :param obj_node_id: identifier for the head node
        :return:
        """
        NDV = Gu.nodes(data=True)
        # Compare and connect nodes in left partition and right partition
        # N.b. there could be zero connections in case of mismatch (False Caption for e.g.)
        is_head_node = lambda x: obj_node_id in x
        Gs_head_nodes = sorted(list(filter(is_head_node, ls)))
        Gt_head_nodes = sorted(list(filter(is_head_node, rs)))
        logger.debug(f'Number of head nodes in Gs (text graph) = {len(Gs_head_nodes)}')
        logger.debug(f'Number of head nodes in Gt (image graph) = {len(Gt_head_nodes)}')
        matching_pairs = []  # Holds the (s,t) matching pairs
        unmatched_pairs = []  # Holds the (s) source nodes that did not match
        for gs_head_node in Gs_head_nodes:
            print(f'Matching source head node: {NDV[gs_head_node]}')
            gs_hn_val = NDV[gs_head_node]['val']
            gs_hn_graph, gs_hn_doc = self.clevr_parser.parse(gs_hn_val)
            gs_hn_span = gs_hn_doc.ents[0]
            matching_grounding_node = None
            # Try matching the src head node to one of the available grounding objects
            for gt_head_node in Gt_head_nodes:
                # gt = nx.subgraph(Gt, gt_head_node)
                # is_equal = __is_equal_head_nodes(gs, gt)
                gt_hn_val = NDV[gt_head_node]['val']
                if gs_hn_val == gt_hn_val:
                    matching_grounding_node = gt_head_node
                    break
                gt_hn_graph, gt_hn_doc = self.clevr_parser.parse(gt_hn_val)
                gt_hn_span = gt_hn_doc.ents[0]
                # Compare <Z>
                attr1 = gs_hn_span._.size if gs_hn_span._.has_size else None
                attr2 = gt_hn_span._.size
                if attr1 is not None and len(attr1) > 0:
                    _is_equal_attr = self.clevr_parser.entity_recognizer.is_equal_size(attr1, attr2)
                    if not _is_equal_attr:
                        continue
                # Compare <C>
                attr1 = gs_hn_span._.color if gs_hn_span._.has_color else None
                attr2 = gt_hn_span._.color
                if attr1 and (attr1.text != attr2.text):
                    # Color is stipulated for source, but doesn't match target
                    continue
                # Compare <M>
                attr1 = gs_hn_span._.material if gs_hn_span._.has_material else None
                attr2 = gt_hn_span._.material
                if attr1 is not None and len(attr1) > 0:
                    _is_equal_attr = self.clevr_parser.entity_recognizer.is_equal_material(attr1, attr2)
                    if not _is_equal_attr:
                        continue
                # Compare <S>
                attr1 = gs_hn_span._.shape if gs_hn_span._.has_shape else None
                attr2 = gt_hn_span._.shape
                if attr1 is not None and len(attr1) > 0:
                    _is_equal_attr = self.clevr_parser.entity_recognizer.is_equal_shape(attr1, attr2)
                    if not _is_equal_attr:
                        continue
                # Found Grounding Node Match
                matching_grounding_node = gt_head_node
                break

            if matching_grounding_node:
                matching_pairs.append((gs_head_node, matching_grounding_node))
            else:
                unmatched_pairs.append(gs_head_node)

        logger.info(f'\tNumber of matching pairs (gs,gt) found = {len(matching_pairs)}'
                    f'\n\tNumber of unmatched pairs (gs) found = {len(unmatched_pairs)}')
        logger.info(f'matching_pairs = {matching_pairs}')

        return matching_pairs, unmatched_pairs

    @classmethod
    def get_nx_graph_edge_indices(cls, G: nx.Graph, nodelist=None):
        """
        Returns edge_indices in a Graph in sparse COO format from a given graph
        :param G: A Graph
        :param nodelist: get edge_index of subgraph of G with given nodelist
        :return: edge_index in [row, col] structure
        """
        try:
            import torch
            from collections import Counter
        except ImportError as ie:
            logger.error(f'{ie}: Could not import torch')

        adj = nx.to_scipy_sparse_matrix(G, nodelist=nodelist).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        return edge_index

    def get_pyg_data_from_nx(self, G: nx.Graph, doc, label=None, pos=None,
                             embd_dim=96, embedding_type=None, is_cuda=False, **kwargs):
        """
        Creates a `torch_geometric.Data` data from G
        :param G:
        :param pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        :return: :Data:
        """
        try:
            import torch
            import torch_geometric
            from torch_geometric.data import Data
        except ImportError as ie:
            logger.error(f'{ie}')

        X = self.get_node_feature_matrix(G, doc, embd_dim=embd_dim,
                                         as_torch=True, is_cuda=is_cuda)
        edge_index = self.get_nx_graph_edge_indices(G)
        edge_attr = self.get_edge_attr_feature_matrix(G, doc, embd_dim=embd_dim,
                                                      as_torch=True, is_cuda=is_cuda)
        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=label)
        if is_cuda and torch.cuda.is_available():
            device = 'cuda'
            data = data.to(device)
        return data

    def get_pyg_pair_data_from_nx(self, Gs: nx.Graph, s_doc, Gt:nx.Graph, t_doc, label=None, pos=None,
                             embd_dim=96, embedding_type=None, is_cuda=False, **kwargs):
        """
        Creates a pair_data where each pair_data sample contains (data_s, data_t) from Gs, Gt
        :param Gs: source graph
        :param Gt: target graph
        :return: Pair data :Data:
        """
        data_s = self.get_pyg_data_from_nx(Gs, s_doc, label=label, embd_dim=embd_dim, is_cuda=is_cuda)
        data_t = self.get_pyg_data_from_nx(Gt, t_doc, label=label, embd_dim=embd_dim, is_cuda=is_cuda)
        x_s, ei_s, ea_s = data_s.x, data_s.edge_index, data_s.edge_attr
        x_t, ei_t, ea_t = data_t.x, data_t.edge_index, data_t.edge_attr
        pair_data = PairData(edge_index_s=ei_s, edge_attr_s=ea_s, x_s=x_s,
                             edge_index_t=ei_t, edge_attr_t=ea_t, x_t=x_t)
        if is_cuda and torch.cuda.is_available():
            device = 'cuda'
            pair_data = pair_data.to(device)

        return pair_data

    def get_pyg_datalist_from_nx(self, graphs:List[nx.Graph], docs, labels=None, poslist=None, **kwargs)-> List:
        """
        Creates and returns a PyG datalist from a list of graphs and corresponding Spacy `Doc`objects
        :param Gss:
        :param docs:
        :param labels:
        :return: List[Data]
        """
        datalist = []
        for i, (G, doc) in enumerate(zip(graphs, docs)):
            label = labels[i] if labels else None
            pos = poslist[i] if poslist else None
            data = self.get_pyg_data_from_nx(G, doc, label, pos, **kwargs)
            datalist.append(data)

        return datalist

    def get_pyg_pair_datalist_from_nx(self, Gss:List[nx.Graph], s_docs, Gts:List[nx.Graph], t_docs,
                                      labels=None, **kwargs)-> List:
        """
        Creates and returns a PyG paired datalist from a list of source, taget graphs and corresponding
        To create a Batch from pair_datalist, use:
        ```batch = Batch.from_data_list(pair_datalist, follow_batch=['x_s', 'x_t']) ```
        Spacy `Doc`objects
        :return: List[Data]
        """
        datalist = []
        for i, (Gs, s_doc, Gt, t_doc) in enumerate(zip(Gss, s_docs, Gts, t_docs)):
            label = labels[i] if labels else None
            #pos = poslist[i] if poslist else None
            pair_data = self.get_pyg_pair_data_from_nx(Gs, s_doc, Gt, t_doc, **kwargs)
            datalist.append(pair_data)

        return datalist

    def get_edge_attr_feature_matrix(self, G:nx.MultiGraph, doc,
                                     embd_dim=96, embedding_type=None, **kwargs):
        """ Edge feature matrix wish shape [num_edges, edge_feat_dim]"""
        assert G is not None
        EDV, EV = G.edges(data=True), G.edges(data=False)
        E, M = len(EDV), embd_dim
        token2vec = {}
        # RS Hack: token pos not delineated, duplicates are overridden
        for token in doc:
            token2vec[token.text] = token.vector
        feat_mats = []
        for i, (_, _, feat_dict) in enumerate(EDV):
            for key, value in feat_dict.items():
                v_embd = token2vec[value]
                logger.debug(f"value = {value}\n v_embd = {v_embd}")
                feat_mats.append(v_embd)
        #unique_values = set(reduce(lambda x1, x2: x1 + x2, data.values))
        if len(feat_mats) > 1:
            feat_mats = reduce(lambda a, b: np.vstack((a, b)), feat_mats)
        else:
            feat_mats = feat_mats[0]
        if kwargs.get('as_torch'):
            feat_mats = torch.from_numpy(feat_mats).float()
            if kwargs.get('is_cuda'):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                feat_mats = feat_mats.to(device)
        assert feat_mats.shape == (E, M)
        return feat_mats

    def get_node_feature_matrix(self, G:nx.MultiGraph, doc, embd_dim=96,
                                embedding_type=None, **kwargs):
        """
        Returns X with shape [num_nodes, node_feat_dim]
        """
        assert G is not None
        NDV = G.nodes(data=True); NV = G.nodes(data=False)
        _is_head_node = lambda x: 'obj' in x
        head_nodes = list(filter(_is_head_node, NV))
        objs = self.clevr_parser.filter_clevr_objs(doc.ents)
        N, M = len(NDV), embd_dim
        feat_mats = []
        for i, entity in enumerate(objs):
            if entity.label_ not in ('CLEVR_OBJS', 'CLEVR_OBJ'):
                continue
            ent_mat = self.clevr_parser.get_clevr_entity_matrix_embedding(entity, dim=96, include_obj_node_emd=True)
            feat_mats.append(ent_mat)
            head_node = G.nodes.get(head_nodes[i])
            pos = head_node.get('pos')  # pos = (x,y,z): Tuple[float]
            # TODO: what's the best way to encode this pos in the feat_mats?
        if len(feat_mats) > 1:
            feat_mats = reduce(lambda a, b: np.vstack((a, b)), feat_mats)
        else:
            feat_mats = feat_mats[0]

        assert feat_mats.shape == (N, M)
        if kwargs.get('as_torch'):
            feat_mats = torch.from_numpy(feat_mats).float()
            if kwargs.get('is_cuda'):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                feat_mats = feat_mats.to(device)

        return feat_mats

    def get_embeddings(self, G: nx.MultiGraph, doc, embd_dim=96,
                       embedding_type=None, **kwargs) -> np.ndarray:
        """
        Example:
        Text: "There is a green metal block; the tiny metal thing is to the left of it"
        Gs -> ['obj', '<C>', '<M>', '<S>', 'obj2', '<Z2>', '<M2>', '<S2>']
        doc.ents -> (green metal block, tiny metal thing)

        NodeDataView({'obj': {'label': 'CLEVR_OBJ', 'val': 'green metal block'},
                        '<C>': {'label': 'color', 'val': 'green'},
                        '<M>': {'label': 'material', 'val': 'metal'},
                        '<S>': {'label': 'shape', 'val': 'block'},
                      'obj2': {'label': 'CLEVR_OBJ', 'val': 'tiny metal thing'},
                            '<Z2>': {'label': 'size', 'val': 'tiny'},
                            '<M2>': {'label': 'material', 'val': 'metal'},
                            '<S2>': {'label': 'shape', 'val': 'thing'}})

        :param G: MultiGraph containing all CLEVR nodes
        :param doc: Spacy Doc
        :param embd_dim:
        :param embedding_type:
        :return: A feature vector matrix corresponding to the value of the Graph of size (N * M) where N is the number
        of nodes in the graph and M corresopnds to the embd_sz (default = 96)
        Note: the head_node ('obj' node) will be a mean of all attrs vecs (of embd_sz)
        """
        import warnings
        warnings.warn("Deprecated: use `get_node_feature_matrix` instead",
                DeprecationWarning, stacklevel=2)
        assert G is not None
        NDV = G.nodes(data=True)
        NV = G.nodes(data=False)
        _is_head_node = lambda x: 'obj' in x
        head_nodes = list(filter(_is_head_node, NV))
        objs = self.clevr_parser.filter_clevr_objs(doc.ents)

        N = len(NDV)
        M = embd_dim
        feat_mats = []
        for i, entity in enumerate(objs):
            if entity.label_ not in ('CLEVR_OBJS', 'CLEVR_OBJ'):
                continue
            ent_mat = self.clevr_parser.get_clevr_entity_matrix_embedding(entity, dim=96, include_obj_node_emd=True)
            feat_mats.append(ent_mat)
            head_node = G.nodes.get(head_nodes[i])
            pos = head_node.get('pos')          # pos = (x,y,z): Tuple[float]
            #TODO: what's the best way to encode this pos in the feat_mats?

        if len(feat_mats) > 1:
            feat_mats = reduce(lambda a, b: np.vstack((a, b)), feat_mats)
        else:
            feat_mats = feat_mats[0]

        assert feat_mats.shape == (N, M)

        ## HACK RS: Inject spatial info Add spatial, matching RE, pos if available
        spatial_ents = self.clevr_parser.filter_spatial_re(doc.ents)
        for i, entity in enumerate(spatial_ents):
            ent_vec = entity.vector.reshape(1, -1)   #(1, 96)
            feat_mats = np.vstack((feat_mats, ent_vec))
        matching_ents = self.clevr_parser.filter_matching_re(doc.ents)
        for i, entity in enumerate(matching_ents):
            ent_vec = entity.vector.reshape(1, -1)  # (1, 96)
            feat_mats = np.vstack((feat_mats, ent_vec))
        ## HACK END ########
        # if as_torch:
        #     feat_mat = torch.from_numpy(feat_mat).float().to(device)
        return feat_mats

    # def from_networkx(self, G):
    #     r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    #     :class:`torch_geometric.data.Data` instance.
    #     # Modified from: torch_geometric.utils.convert.py
    #     Args:
    #         G (networkx.Graph or networkx.DiGraph): A networkx graph.
    #     """
    #     try:
    #         import torch_geometric
    #     except ImportError as ie:
    #         logger.error(ie)
    #     vocab = load_vocab(path="../data/reason/clevr_h5/clevr_vocab.json")
    #     q2t = vocab['question_token_to_idx']
    #     # value = q2t[value] if q2t.get(value) else value
    #     G = nx.convert_node_labels_to_integers(G)
    #     #G = G.to_directed() if not nx.is_directed(G) else G
    #     # edge_index = torch.tensor(list(G.edges)).t().contiguous()
    #     edge_index = self.get_nx_graph_edge_indices(G)
    #     data = {}
    #     #data = collections.Counter()
    #     for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
    #         for key, value in feat_dict.items():
    #             if i == 0:
    #                 data[key] = [value]
    #             else:
    #                 if data.get(key):
    #                     data[key] += [value]
    #                 else:
    #                     data[key] = [value]
    #
    #     for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
    #         for key, value in feat_dict.items():
    #             if i == 0:
    #                 data[key] = [value]
    #             else:
    #                 if data.get(key):
    #                     data[key] += [value]
    #                 else:
    #                     data[key] = [value]
    #
    #     #data['edge_index'] = edge_index.view(2, -1)
    #     data['edge_index'] = edge_index
    #     data = torch_geometric.data.Data.from_dict(data)
    #     data.num_nodes = G.number_of_nodes()
    #     return data

    ## End of Functions for Graph Embeddings ##

