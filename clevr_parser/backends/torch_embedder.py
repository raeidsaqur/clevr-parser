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

import logging
from functools import reduce
from typing import List, Tuple

import numpy as np

from .backend import EmbedderBackend
from .spacy_parser import SpacyParser
from ..embedder import Embedder
# from ..utils import *
from ..utils import load_grounding_for_img_idx

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

import networkx as nx

try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
except ImportError as ie:
    logger.error(f"Some required modules couldn't be imported: {ie.name}")

__all__ = ['TorchEmbedder']

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

    # --------------------------- Interface Methods --------------------------------------- #
    def embed_s(self, sentence, *args, **kwargs):
        s = sentence
        Gs, s_doc = kwargs.get('Gs'), kwargs.get('s_doc')
        if not (Gs and s_doc):
            try:
                Gs, s_doc = self.clevr_parser.parse(s, return_doc=True,
                                                    is_directed_graph=True)
            except ValueError as ve:
                logger.error(f"ValueError Encountered: {ve}")
                return None
        return self._embed(Gs, s_doc, *args, **kwargs)

    def embed_t(self, img_idx: int, img_scene_path: str, *args, **kwargs):
        img_scene = kwargs.get('img_scene')
        if img_scene is None:
            img_scene = load_grounding_for_img_idx(img_idx, img_scene_path)
        try:
            Gt, t_doc = self.clevr_parser.get_doc_from_img_scene(img_scene,
                                                is_directed_graph=True)
        except FileNotFoundError as fne:
            logger.error(fne)
            return None
        return self._embed(Gt, t_doc, *args, **kwargs)

    def _embed(self, G, doc, *args, **kwargs):
        X = self.get_node_feature_matrix(G, doc, **kwargs)
        edge_index = self.get_nx_graph_edge_indices(G, **kwargs)
        edge_attr = self.get_edge_attr_feature_matrix(G, doc, **kwargs)
        return X, edge_index, edge_attr

    # --------------------------- Concrete Embedder Specific Methods --------------------------------- #
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
                #logger.debug(f'Source attr node label = {ns_label}')
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
    def get_nx_graph_edge_indices(cls, G: nx.Graph, nodelist=None, **kwargs):
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
        except ImportError as ie:
            logger.error(f'{ie}')

        X = self.get_node_feature_matrix(G, doc, embd_dim=embd_dim,
                                         as_torch=True, is_cuda=is_cuda)
        edge_index = self.get_nx_graph_edge_indices(G)
        edge_attr = self.get_edge_attr_feature_matrix(G, doc, embd_dim=embd_dim,
                                                      as_torch=True, is_cuda=is_cuda)
        # RuntimeError: Edge indices and edge attributes hold a differing number of edges, found torch.Size([2, 1]) and torch.Size([96])
        # data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=label)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=label)
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
                                     embd_dim=96, embedding_type=None, as_torch=True, is_padding_pos=True, **kwargs):
        """ Edge feature matrix wish shape [num_edges, edge_feat_dim]"""
        assert G is not None
        EDV, EV = G.edges(data=True), G.edges(data=False)
        E = len(EDV)
        M = (embd_dim+3) if is_padding_pos else embd_dim
        token2vec = {}
        # RS N.b.: Gt side token 'pos' are captured in node attr, duplicates are overridden
        for token in doc:
            token2vec[token.text] = token.vector
        feat_mats = []
        for i, (_, _, feat_dict) in enumerate(EDV):
            for key, value in feat_dict.items():
                v_embd = token2vec[value]
                v_embd = v_embd.reshape((-1, embd_dim))
                #logger.debug(f"value = {value}\n v_embd = {v_embd}")
                if is_padding_pos:
                    v_embd = np.pad(v_embd, ((0,0), (0, 3)), 'constant', constant_values=(0, 0.0))  # [1, embd_dim+3]
                feat_mats.append(v_embd)
        #unique_values = set(reduce(lambda x1, x2: x1 + x2, data.values))
        if len(feat_mats) > 1:
            feat_mats = reduce(lambda a, b: np.vstack((a, b)), feat_mats)
        else:
            feat_mats = feat_mats[0].reshape((1, -1))
        if as_torch:
            feat_mats = torch.from_numpy(feat_mats).float()
            if kwargs.get('is_cuda'):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                feat_mats = feat_mats.to(device)
        try:
            # Ensure single edges are properly reshaped
            assert feat_mats.shape == (E, M)
        except AssertionError as ae:
            ## RuntimeError: Edge indices and edge attributes hold a differing number of edges,
            logger.debug(f"{ae}\n feat_mats.shape = {feat_mats.shape}")
        return feat_mats

    def get_node_feature_matrix(self, G:nx.MultiGraph, doc, embd_dim=96,
                                embedding_type=None, as_torch=True, is_padding_pos=True, **kwargs):
        """
        Returns X with shape [num_nodes, node_feat_dim]
        """
        assert G is not None
        NDV = G.nodes(data=True); NV = G.nodes(data=False)
        _is_head_node = lambda x: 'obj' in x
        head_nodes = list(filter(_is_head_node, NV))
        objs = self.clevr_parser.filter_clevr_objs(doc.ents)
        N, M = len(NDV), (embd_dim+3) if is_padding_pos else embd_dim
        feat_mats = []
        for i, entity in enumerate(objs):
            if entity.label_ not in ('CLEVR_OBJS', 'CLEVR_OBJ'):
                continue
            ent_mat = self.clevr_parser.get_clevr_entity_matrix_embedding(entity, dim=embd_dim,
                                                                          include_obj_node_emd=True,
                                                                          is_padding_pos=is_padding_pos)
            ent_mat = ent_mat.reshape((-1, embd_dim))
            if is_padding_pos:
                head_node = G.nodes.get(head_nodes[i])
                pos = head_node.get('pos')  # pos = (x,y,z): Tuple[float]
                if not pos:
                    pos = (0.0, 0.0, 0.0)
                pos = np.tile(np.asarray(pos, dtype=float), ent_mat.shape[0]).reshape(-1, 3)
                # np.pad(ent_mat, ((0,0), (0, 3)), 'constant', constant_values=(0, 0.0) )
                ent_mat = np.concatenate((ent_mat, pos), axis=1)        # [n, embd_dim+3]
            feat_mats.append(ent_mat)
        if len(feat_mats) > 1:
            feat_mats = reduce(lambda a, b: np.vstack((a, b)), feat_mats)
        else:
            feat_mats = feat_mats[0]

        assert feat_mats.shape == (N, M)
        if as_torch:
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

        if len(feat_mats) > 1:
            feat_mats = reduce(lambda a, b: np.vstack((a, b)), feat_mats)
        else:
            feat_mats = feat_mats[0]
        assert feat_mats.shape == (N, M)

        spatial_ents = self.clevr_parser.filter_spatial_re(doc.ents)
        for i, entity in enumerate(spatial_ents):
            ent_vec = entity.vector.reshape(1, -1)   #(1, 96)
            feat_mats = np.vstack((feat_mats, ent_vec))
        matching_ents = self.clevr_parser.filter_matching_re(doc.ents)
        for i, entity in enumerate(matching_ents):
            ent_vec = entity.vector.reshape(1, -1)  # (1, 96)
            feat_mats = np.vstack((feat_mats, ent_vec))

        return feat_mats


    ## End of Functions for Graph Embeddings ##

