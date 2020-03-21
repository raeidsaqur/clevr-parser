#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : spacy_parser.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/21/2019
#
# This file is part of PGFM Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

from .. import database
from ..parser import Parser
from .backend import ParserBackend
from .custom_components_clevr import CLEVRObjectRecognizer
from ..utils import *

__all__ = ['SpacyParser']

from functools import reduce
from operator import itemgetter
from typing import List, Dict, Tuple, Sequence
import copy
import logging
logger = logging.getLogger(__name__)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import pygraphviz as pgv
    import networkx as nx
except ImportError as ie:
    logger.error(f"Install NetworkX: {ie.name}")

import numpy as np
np.random.seed(42)
import scipy.sparse as sp

@Parser.register_backend
class SpacyParser(ParserBackend):
    """
    Scene graph parser based on spaCy.
    """

    __identifier__ = 'spacy'

    def __init__(self, model='en'):
        """
        Args:
            model (str): a spec for the spaCy model. (default: en). Please refer to the
            official website of spaCy for a complete list of the available models.
            This option is useful if you are dealing with languages other than English.
        """
        super().__init__()
        self.__model = model
        try:
            import spacy
        except ImportError as e:
            raise ImportError('Spacy backend requires the spaCy library. Install spaCy via pip first.') from e
        try:
            self.__nlp = spacy.load(model)
        except OSError as e:
            raise ImportError('Unable to load the English model. Run `python -m spacy download en` first.') from e

        self.__entity_recognizer = CLEVRObjectRecognizer(self.__nlp)

    @property
    def entity_recognizer(self):
        return self.__entity_recognizer

    @entity_recognizer.setter
    def entity_recognizer(self, er):
        self.__entity_recognizer = er

    @property
    def nlp(self):
        return self.__nlp
    @nlp.setter
    def nlp(self, nlp):
        self.__nlp = nlp

    @property
    def model(self):
        return self.__model
    @model.setter
    def model(self, model):
        self.__model = model


    def parse(self, sentence:str, index=0, filename=None, return_doc=True):
        """
            The spaCy-based parser parse the sentence into scene graphs based on the dependency parsing
            of the sentence by spaCy.

            Returns a nx.MultiGraph and Spacy.doc
        """
        doc = self.__nlp(sentence)
        is_plural = doc._.has_shapes
        if is_plural:
            logger.info(f'{sentence} contains plural, skipping all CLEVR_OBJS as an edge case')
            return None, f"SKIP_img {index}_{filename}"

        graph, en_graphs = self.get_nx_graph_from_doc(doc)

        # N.b. The ordering of doc.ents and graph.nodes should be aligned
        if return_doc:
            return graph, doc
        return graph


    def parse2(self, sentence:str, index=0, filename=None, return_doc=True):
        """
            #TODO: combine entity and Grammar rules for later, `parse` simply does
            CLEVR entity graphs.
            The spaCy-based parser parse the sentence into scene graphs based on the dependency parsing
            of the sentence by spaCy.

            Returns a nx.MultiGraph and Spacy.doc
        """
        doc = self.__nlp(sentence)
        graph, en_graphs = self.get_nx_graph_from_doc(doc)


        # Step 2: determine the relations between objs
        relation_subj = dict()
        for token in doc:
            # E.g., A [woman] is [playing] the piano.
            if token.dep_ == 'nsubj':
                relation_subj[token.head.i] = token.i
            # E.g., A [woman] [playing] the piano...
            elif token.dep_ == 'acl':
                relation_subj[token.i] = token.head.i
            # E.g., The piano is [played] by a [woman].
            elif token.dep_ == 'pobj' and token.head.dep_ == 'agent' and token.head.head.pos_ == 'VERB':
                relation_subj[token.head.head.i] = token.i

        # Step 3: determine the relations.
        relations = list()
        filtered_relations = list()
        fake_noun_marks = set()

        for object in objects:
            # Again, the subjects and the objects are represented by their position.
            relation = None
            # small red rubber cylinder is behind a large brown metal sphere

            if relation is not None:
                relations.append(relation)

        """
        # Apply the `fake_noun_marks`.
        entities = [e for e, ec in zip(entities, entity_chunks) if ec.root.i not in fake_noun_marks]
        entity_chunks = [ec for ec in entity_chunks if ec.root.i not in fake_noun_marks]
        
        for relation in relations:
            # Use a helper function to map the subj/obj represented by the position
            # back to one of the entity nodes.
            relation['subject'] = self.__locate_noun(entity_chunks, relation['subject'])
            relation['object'] = self.__locate_noun(entity_chunks, relation['object'])
            if relation['subject'] != None and relation['object'] != None:
                filtered_relations.append(relation)
        """

        # N.b. The ordering of doc.ents and graph.nodes should be aligned
        if return_doc:
            return graph, doc
        return graph

    def get_clevr_text_vector_embedding(self, text, ent_vec_size=384, embedding_type=None):
        """
        Takes a text input and returns the feature vector X
        :param text: Caption or Question (any NL input)
        :param ent_vec_size: size of each entity.
        :param embedding_type: GloVe, BERT, GPT etc.
        :return:
        """
        assert text is not None
        G_text, doc = self.parse(text, return_doc=True)
        if G_text is None and 'SKIP' in doc:
            logger.info(f'{text} contains plural (i.e. label CLEVR_OBJS')
            return None, f"SKIP_{text}"

        doc_emd = self.get_clevr_doc_vector_embedding(doc, ent_vec_size=ent_vec_size, embedding_type=embedding_type)

        return G_text, doc_emd

    def get_clevr_doc_vector_embedding(self, doc,
                                        attr_vec_size=96,
                                        ent_vec_size=384,
                                        include_obj_node_emd=True,
                                        embedding_type=None):
        """
        Embedding = [<Gs-obj>, <Z>, <C>, <M>, <S>]
        To keep dim d constant, we pad missing attrs with [0]*attr_vec_size
        <Gs-obj> is just a copy of [<Z>, <C>, <M>, <S>].
        So total ent_vec_size = (attr_vec_size*4) * 2

        :param doc: A spacy Doc
        :param ent_vec_size: embedding vector size of each clevr entity
        :param include_obj_node_emd: if True, a wrapper obj node "obj" is prepended to the entity vector embedding
        :param embedding_type: one-hot, bag_of_words, GloVe etc.
        :return: vector embedding for all the clevr entitites in a doc
        """
        assert doc is not None
        entities = doc.ents
        embed_sz = len(entities) * ent_vec_size
        if include_obj_node_emd:
            embed_sz *= 2          # 2x size due to tiling of obj node vector
        doc_vector = np.zeros((embed_sz,), dtype=np.float32).reshape((1, -1))
        ent_vecs = []
        for entity in entities:
            if entity.label_ != 'CLEVR_OBJ':
                continue
            ent_vec = self.get_clevr_entity_vector_embedding(entity, ent_vec_size, include_obj_node_emd, embedding_type)
            ent_vecs.append(ent_vec)

        doc_vector = np.hstack(tuple(ent_vecs)).reshape((1, -1))

        assert doc_vector.shape[1] == embed_sz
        return doc_vector

    def get_clevr_entity_matrix_embedding(self, entity, dim=96, include_obj_node_emd=True, embedding_type=None):
        """
        Atomic function for generating matrix embedding from a doc entity:

        :param entity: A spacy Doc.entity
        :param dim: the dim of the embd matrix, default = 96, the same dim as each attr node
        :param include_obj_node_emd: if True, a wrapper obj node "obj" is prepended to the entity vector embedding
        :param embedding_type: one-hot, bag_of_words, GloVe etc.
        :return: an N by M embedding matrix, where M = dim and N is the num of nodes of a generated graph of the entity
        """
        label = entity.label_
        if label is None or label != "CLEVR_OBJ":
            raise TypeError("The entity must be a CLEVR_OBJ entity")

        # poss = []
        # embds = []
        embds_poss = []
        for token in entity:
            _v, pos = self.get_attr_token_vector_embedding(token, size=dim, embedding_type=embedding_type)
            # poss.append(pos)
            # embds.append(_v)
            embds_poss.append((_v, pos))
        #embds = reduce(lambda a,b: np.vstack((a,b)), embds) if len(embds) > 1 else embds[0]
        embds_poss.sort(key=itemgetter(1))
        embds = list(map(lambda x: x[0], embds_poss))
        embds = np.array(embds, dtype=np.float32).squeeze()
        #embds = embds[poss]
        obj_embd = np.mean(embds,axis=0)
        embds = np.vstack((obj_embd, embds))

        assert embds.shape[-1] == dim

        return embds


    def get_clevr_entity_vector_embedding(self, entity, size=384, include_obj_node_emd=True, embedding_type=None):
        """
        Atomic function for generating vector embedding from a doc entity:

        :param entity: A spacy Doc.entity
        :param size: N.b., the token embedding size will be ent_size / 4 (rank-4 is the full tensor)
        :param include_obj_node_emd: if True, a wrapper obj node "obj" is prepended to the entity vector embedding
        :param embedding_type: one-hot, bag_of_words, GloVe etc.
        :return: a uniform sized entity vector embedding, if it's not a full-rank attr tensor,
        then the entity vector needs to be padded. For e.g., "red thing" -> "<C> <S>" with missing
        <Z>, <M> attrs, in which case, <Z> <C> <M> <S> entity embedding will have <Z> <M> padded
        """
                                        #default_entity_vector_dim = 384
        token_sz= int(size / 4)         #default = 384/4 = 96
        label = entity.label_
        if label is None or label != "CLEVR_OBJ":
            raise TypeError("The entity must be a CLEVR_OBJ entity")

        # missing attr tokens are represented with token_sz * 0.0
        ent_vector = np.zeros((size,), dtype=np.float32).reshape((1, -1))
        for token in entity:
            _v, pos = self.get_attr_token_vector_embedding(token, size=token_sz, embedding_type=embedding_type)
            s_idx = pos * token_sz
            e_idx = s_idx + token_sz
            ent_vector[:, s_idx: e_idx] = _v

        if include_obj_node_emd:
            # Duplicate and prepend the env_vector
            ent_vector = np.tile(ent_vector, 2)

        return ent_vector

    def _get_attr_token_pos(self, token):
        """
        :param token: <Z> <C> <M> <S>
        :return: a pos int in the range (0, 3) based on the relative ordering
        """
        t = token  # N.b. the pipeline must have clevr 'ent_recognizer' added with extensions
        pos = 0
        if t._.is_size:
            pos = 0
        elif t._.is_color:
            pos = 1
        elif t._.is_material:
            pos = 2
        elif t._.is_shape:
            pos = 3
        else:
            # fon anything else, place it after the attr pos
            pos = 4

        return int(pos)

    def get_attr_token_vector_embedding(self, token, size=96, embedding_type=None):
        """

        :param token: A Spacy token belonging to a (Doc) entity and awith clevr entity recognizer
        baked in, i.e., the clevr extensions are (presumed) available
        :param size: The dimension of the embedding vector
        :param embedding_type:
        :return: returns a token embedding vector of specified size (default=96) and relative position
        in an entity [Z C M S] embedding vector
        """
        # default_token_vector_dim = 96
        # size = size if size is not None else default_token_vector_dim
        if embedding_type is None:
            # Use the default embedding type
            vector = token.vector.reshape(1, -1)

        pos = self._get_attr_token_pos(token)

        return vector, pos


    @staticmethod
    def get_attr_node_from_token(token, ent_num=0):
        """
        :param token: A Spacy token belonging to a (Doc) entity
        :param ent_num: the entity number, default 0, used to assign markers
        like '<Z>' (default), or <Z2> (for 2obj)
        :return: a nx graph node construction structure
        """
        assert token is not None
        node_keys = ('label', 'val')
        # Reformat in nx.graph node construct structure
        _n_fn = lambda s, a, t: tuple((s, dict(zip(node_keys, (a, t.text)))))
        t = token # N.b. the pipeline must have clevr 'ent_recognizer' added with extensions
        if t._.is_size:
            s = "<Z>" if ent_num <= 1 else f"<Z{ent_num}>"
            node = tuple(_n_fn(s, 'size', t))
        elif t._.is_color:
            s = "<C>" if ent_num <= 1 else f"<C{ent_num}>"
            node = _n_fn(s, 'color', t)
        elif t._.is_material:
            s = "<M>" if ent_num <= 1 else f"<M{ent_num}>"
            node = _n_fn(s, 'material', t)
            # node = ('M', dict(zip(node_keys, ('material', t.text))))
        if t._.is_shape:
            s = "<S>" if ent_num <= 1 else f"<S{ent_num}>"
            node = _n_fn(s, 'shape', t)

        return node

    def get_pos_from_img_scene(self, scene, *args, **kwargs):
        scene_img_idx = scene["image_index"]
        scene_img_fn = scene["image_filename"]
        clevr_objs = scene['objects']
        nco = len(clevr_objs)
        assert nco <= 10

        p = lambda o: tuple(o['position'])  # (x, y, z) co-ordinates
        pos = list(map(p, clevr_objs))

        return pos

    def get_caption_from_img_scene(self, scene, *args, **kwargs):

        scene_img_idx = scene["image_index"]
        scene_img_fn = scene["image_filename"]
        clevr_objs = scene['objects']
        nco = len(clevr_objs)
        assert nco <= 10
        if nco == 0:
            logger.warning(f"Scene derendering appears to have failed on {scene_img_idx}: {scene_img_fn}"
                           f"\nThe derenderer failed to produce any proposal for this scene image."
                           f"\nSkipping this scene image from data")
            #return f"SKIP_{scene_img_idx}_{scene_img_fn}"
            return None

        f = lambda o: " ".join([o['size'], o['color'], o['material'], o['shape']])
        concat = lambda x, y: x + ", " + y
        caption = reduce(concat, map(f, clevr_objs))  # skeletal scene caption without pos, rel
        # p = lambda o: tuple(o['position'])  # (x, y, z) co-ordinates
        # pos = list(map(p, clevr_objs))

        return caption

    def get_doc_from_img_scene(self, scene, *args, **kwargs):
        """
        TODO: not utilizing the position info in parsed img scene
        :param scene:
        :param args:
        :param kwargs:
        :return:
        """
        scene_img_idx = scene["image_index"]
        scene_img_fn = scene["image_filename"]
        caption = self.get_caption_from_img_scene(scene, *args, **kwargs)
        if caption is None:
            return None, f"SKIP_{scene_img_idx}_{scene_img_fn}"

        graph, doc = self.parse(caption, return_doc=True)
        return graph, doc


    @classmethod
    #@trace
    def get_graph_from_entity(cls, entity, ent_num=0,
                              is_directed_graph=False,
                              is_attr_name_node_label = False,
                              head_node_prefix=None,
                              hnode_sz=1200, anode_sz=700,
                              hnode_col='tab:blue', anode_col='tab:red',
                              is_return_list= False,
                              is_debug=False):
        """
        The atomic graph constructor.
        :param entity: atomic CLEVR Object
        :param ent_num: the id of the object in context of the full graph
        :param is_attr_name_node_label:
        :param head_node_prefix:
        :param hnode_sz:
        :param anode_sz:
        :param hnode_col:
        :param anode_col:
        :param is_return_list:
        :param is_debug:
        :return:
        """
        obj_vals = (entity.label_, entity.text)
        node_keys = ('label', 'val')
        d = dict(zip(node_keys, obj_vals))
        head_node_id = "obj" if ent_num <= 1 else f"obj{ent_num}"
        if head_node_prefix and (head_node_prefix not in head_node_id):
            head_node_id = f"{head_node_prefix}-{head_node_id}"

        nodelist = [tuple((head_node_id, d))]

        _n_fn = lambda s, a, t: tuple((s, dict(zip(node_keys, (a, t.text)))))
        for t in entity:
            _node = cls.get_attr_node_from_token(t, ent_num)
            nodelist.append(_node)

        # Node Labels
        if is_attr_name_node_label:
            labels = dict(map(lambda x: (x[0], x[1]['label']), nodelist))
        else:
            #print(nodelist[0])
            labels = dict(map(lambda x: (x[0], x[1]['label']), [nodelist[0]]))
            a_labels = dict(map(lambda x: (x[0], x[1]['val']), nodelist[1:]))
            labels.update(a_labels)

        # Edge List & Labels:
        edgelist = []
        edge_labels = {}  # edge_labels = {(u, v): d for u, v, d in G.edges(data=True)}
        _e_fn = lambda x: tuple((head_node_id, x[0], {x[1]['label']: x[1]['val']}))
        for i, node in enumerate(nodelist):
            if node[0] == head_node_id:
                continue
            edge = _e_fn(node)
            edgelist.append(edge)
            edge_label = f"{node[0]}:{node[1]['label']}"
            edge_labels.update({(head_node_id, node[0]): edge_label})

        G = nx.MultiDiGraph() if is_directed_graph else nx.MultiGraph()
        G.add_nodes_from(nodelist)
        G.add_edges_from(edgelist)

        if is_debug:
            print(f'node_labels = {labels}')
            print(f"edge_labels = {edge_labels}")
            print(f"G.nodes = {G.nodes(data=True)}")
            print(f"G.edges = {G.edges}")
            print(f"G.adj = {G._adj}")

        l = len(nodelist) - 1
        nsz = [hnode_sz]
        nsz.extend([anode_sz] * l)
        nc = [hnode_col]
        nc.extend([anode_col] * l)
        if is_debug:
            print('\n')
            print(f"nsz = {nsz}")
            print(f"nc = {nc}\n")

        if is_return_list:
            [G, nodelist, labels, edgelist, edge_labels, nsz, nc]
        return G, nodelist, labels, edgelist, edge_labels, nsz, nc

    def get_docs_from_nx_graph(cls, G:nx.Graph) -> List:
        nodes: nx.NodeDataView = G.nodes(data=True)
        # clevr_obj_nodes: List[Tuple] = list(filter(lambda n: n[1]['label'] == 'CLEVR_OBJ', nodes))
        clevr_spans: List[str] = list(map(lambda x: x[1]['val'], filter(lambda n: n[1]['label'] == 'CLEVR_OBJ', nodes)))
        # E.g. : ['small red rubber cylinder', 'large brown metal sphere']
        nco = len(clevr_spans)
        assert nco <= 10
        if nco == 0:
            logger.warning(f"No CLEVR_OBJ found in {clevr_spans}")
            return None
        _docs = []
        for cs in clevr_spans:
            _, _doc = cls.parse(cs)
            _docs.append(_doc)

        return _docs


    @classmethod
    def get_nx_graph_from_doc(cls, doc, head_node_prefix=None):
        """
        :param doc: doc obtained upon self.nlp(caption|text) contains doc.entities as clevr objs
        :return: a composed NX graph of all clevr objects along with pertinent info in en_graphs
        """
        assert doc.ents is not None
        nco = len(doc.ents)
        assert nco <= 10        # max number of clevr entities in one scene

        en_graph_keys = list(range(1, nco + 1))
        en_graph_vals = ['graph', 'nodelist', 'labels', 'edgelist', 'edge_labels', 'nsz', 'nc']
        en_graphs = dict.fromkeys(en_graph_keys)

        graphs = []  # list of all graphs corresponding to each entity
        for i, en in enumerate(doc.ents):
            en_graph_key = en_graph_keys[i]
            #print(f"Processing graph {en_graph_key} ... ")
            _g = cls.get_graph_from_entity(en, head_node_prefix=head_node_prefix, ent_num=i + 1, is_return_list=True)
            if isinstance(_g[0], nx.Graph):
                graphs.append(_g[0])
            assert len(en_graph_vals) == len(_g)
            en_graph_dict = dict(zip(en_graph_vals, _g))
            en_graphs[en_graph_key] = en_graph_dict

        ## Multi-Obj case ##
        if len(graphs) > 0:
            G = nx.compose_all(graphs)
        else:
            raise ValueError("0 graphs could be parsed from the given spacy.Doc")

        return G, en_graphs

    # @classmethod
    @trace
    def draw_clevr_img_scene_graph(self, scene,
                                   hnode_sz=1200, anode_sz=700,
                                   hnode_col='tab:blue', anode_col='tab:red',
                                   font_size=12,
                                   show_edge_labels=True,
                                   plot_box=False,
                                   save_file_path=None,
                                   debug=False):
        """
        Steps:
        1. generate (canonical) caption from the image scene for all the objects
        2. parser.parse(caption) -> graph, doc
        3. call parser.draw_clevr_obj_graph() # same used for text scene graph.

        Issues:
        1. Need to encode the positional information in image scene
        """
        graph, doc = self.get_doc_from_img_scene(scene)

        if graph is None and doc.contains("SKIP"):
            return None

        kwargs = {
            'hnode_sz': hnode_sz,
            'anode_sz': anode_sz,
            'hnode_col': hnode_col,
            'anode_col': anode_col,
            'font_size': font_size,
            'show_edge_labels': show_edge_labels,
            'plot_box': plot_box,
            'save_file_path': save_file_path,
            'debug': debug
        }
        G = self.__class__.draw_clevr_obj_graph(graph, doc, **kwargs)
        return G

    @classmethod
    def draw_clevr_obj_graph(cls, text_scene_graph, doc,
                             hnode_sz=1200, anode_sz=700,
                             hnode_col='tab:blue', anode_col='tab:red',
                             font_size=12,
                             show_edge_labels=True,
                             plot_box=False,
                             save_file_path=None,
                             debug=False):
        ax_title = f"{doc}"
        G, en_graphs = cls.get_nx_graph_from_doc(doc)
        G = cls.draw_graph(G, en_graphs, ax_title=ax_title)
        return G

    @classmethod
    def draw_graph(cls, G, en_graphs,
                   hnode_sz=1200, anode_sz=700,
                   hnode_col='tab:blue', anode_col='tab:red',
                   font_size=12,
                   show_edge_labels=True,
                   plot_box=False,
                   save_file_path=None,
                   ax_title=None,
                   debug=False):

        ### Nodes
        NDV = G.nodes(data=True)
        NV = G.nodes(data=False)
        _is_head_node = lambda x: 'obj' in x
        _is_attr_node = lambda x: 'obj' not in x
        head_nodes = list(filter(_is_head_node, NV))
        attr_nodes = list(filter(_is_attr_node, NV))
        assert len(NDV) == len(head_nodes) + len(attr_nodes)
        pos = nx.layout.bipartite_layout(G, nodes=head_nodes)
        #pos = nx.layout.spectral_layout(G)
        pos = nx.spring_layout(G, pos=pos, fixed=attr_nodes)
        #pos = nx.spring_layout(G)  # Get node positions

        # Create position copies for shadows, and shift shadows
        # See: https://gist.github.com/jg-you/144a35013acba010054a2cc4a93b07c7
        pos_shadow = copy.deepcopy(pos)
        shift_amount = 0.001
        for idx in pos_shadow:
            pos_shadow[idx][0] += shift_amount
            pos_shadow[idx][1] -= shift_amount

        nsz = [hnode_sz if 'obj' in node else anode_sz for node in G.nodes]
        # nsz2 = list(map(lambda node: hnode_sz if 'obj' in node else anode_sz, G.nodes))
        nc = [hnode_col if 'obj' in node else anode_col for node in G.nodes]

        if debug:
            print(G.nodes(data=True))

        #### Node Labels: Label head nodes as obj or obj{i}, and attr nodes with their values:
        _label = lambda node: node[1]['val'] if 'obj' not in node[0] else node[0]
        _labels = list(map(_label, G.nodes(data=True)))
        labels = dict(zip(list(G.nodes), _labels))

        ### Edges
        #### Edge Labels
        edge_labels = {}
        for k, v in en_graphs.items():
            edge_labels.update(v['edge_labels'])
        if debug:
            print(edge_labels)

        #### Add <R>, <R2> etc. edge between nodes
        head_nodes = []
        for i, node in enumerate(list(G.nodes(data=False))):
            #if i % 5 == 0:
            if 'obj' in node:
                head_nodes.append(node)
        print(f"head_nodes = {head_nodes}")
        if len(head_nodes) > 1:
            # ToDo: there needs to be connection among all head node permutations
            for i, h_node in enumerate(head_nodes):
                if i == 0:
                    continue
                h = head_nodes[i - 1]
                t = h_node
                # TODO: Relations should be order invariant, remove i+1
                key = "<R>" if i <= 1 else f"<R{i + 1}>"
                G.add_edges_from([(h, t, {key: "tbd"})])
                edge_labels.update({(h, t): key})
                
        # G.add_edges_from([('obj', 'obj2', {'<R>': 'tbd'})])
        # edge_labels.update({('obj', 'obj2'): "<R>"})

        edgelist = G.edges(data=True)

        ## Draw ##

        # Render (MatPlotlib)
        plt.axis('on' if plot_box == True else "off")
        # fig, axs = plt.subplots(1, 2)
        # axs[1].set_title(f"{doc}")
        fig, ax = plt.subplots(1, 1)
        ax.set_title(ax_title)

        nx.draw_networkx_nodes(G, pos, node_size=nsz, node_color=nc)
        nx.draw_networkx_nodes(G, pos_shadow, node_size=nsz, node_color='k', alpha=0.2)

        nx.draw_networkx_edges(G, pos, edgelist=edgelist)
        # nx.draw_networkx_edges(G, pos, edgelist=G.edges(data=True))
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, font_color='k', font_family='sans-serif')
        if show_edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)

        if save_file_path is not None:
            plt.savefig(save_file_path)
        # if pygraphviz_enabled:
        #   nx.write_dot(G, 'file.dot')
        plt.show()

        return G

    @classmethod
    def plot_graph_graphviz(cls, G):

        try:
            import random
            from networkx.drawing.nx_agraph import graphviz_layout
            from networkx.algorithms.isomorphism.isomorph import (
                graph_could_be_isomorphic as isomorphic,
            )
            from networkx.generators.atlas import graph_atlas_g
        except ImportError as ie:
            logger.error(f"Install pygraphviz and graphviz: {ie}")

        #print(f"graph has {nx.number_of_nodes(G)} nodes with {nx.number_of_edges(G)} edges")
        #print(nx.number_strongly_connected_components(G), "connected components")

        plt.figure(1, figsize=(8, 8))
        # layout graphs with positions using graphviz neato
        pos = graphviz_layout(G, prog="neato")
        # color nodes the same in each connected subgraph
        C = (G.subgraph(c) for c in nx.strongly_connected_components(G))
        for g in C:
            c = [random.random()] * nx.number_of_nodes(g)  # random color...
            nx.draw(g, pos, node_size=40, node_color=c, vmin=0.0, vmax=1.0, with_labels=False)
        plt.show()

    @classmethod
    def plot_graph(cls, G, nodelist, labels, edgelist, edge_labels, nsz, nc, font_size=12, show_edge_labels=True):
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=nsz, node_color=nc)
        nx.draw_networkx_edges(G, pos, edgelist=edgelist)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, font_color='k', font_family='sans-serif')
        if show_edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)

    @classmethod
    def plot_entity_graph_dict(cls, entity_graph, font_size=12, show_edge_labels=True):
        en_graph_vals = ['graph', 'nodelist', 'labels', 'edgelist', 'edge_labels', 'nsz', 'nc']
        G, nodelist, labels, edgelist, edge_labels, nsz, nc = list(map(lambda x: entity_graph[x], en_graph_vals))

        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=nsz, node_color=nc)
        nx.draw_networkx_edges(G, pos, edgelist=edgelist)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, font_color='k', font_family='sans-serif')
        if show_edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)

    @classmethod
    def visualize(cls, doc, dep=False):
        try:
            from spacy import displacy
            displacy.serve(doc, style='ent', options={'compact': True})
            if dep:
                displacy.serve(doc, style='dep', options={'compact': True})
        except ImportError as ie:
            logger.error("Could not import displacy for visualization")

    @staticmethod
    def __locate_noun(chunks, i):
        for j, c in enumerate(chunks):
            if c.start <= i < c.end:
                return j
        return None

    # @classmethod
    # def get_graph_edge_labels(cls, G):
    #     # Edge List & Labels:
    #     edgelist = []
    #     edge_labels = {}  # edge_labels = {(u, v): d for u, v, d in G.edges(data=True)}
    #     # _e_fn = lambda x: tuple((head_node_id, x[0], {x[1]['label']: x[1]['val']}))
    #     # for i, node in enumerate(nodelist):
    #     #     if node[0] == head_node_id:
    #     #         continue
    #     #     edge = _e_fn(node)
    #     #     edgelist.append(edge)
    #     #     edge_label = f"{node[0]}:{node[1]['label']}"
    #     #     edge_labels.update({(head_node_id, node[0]): edge_label})

