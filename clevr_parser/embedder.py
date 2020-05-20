#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : embedder.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/21/2019
#
# This file is part of CLEVR Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

__all__ = ['Embedder', 'get_default_embedder', 'embed_s', 'embed_t']

def _load_backends():
    from . import backends

class Embedder(object):
    """
    Example::
    >>> embedder = Embedder(backend, **init_kwargs)
    >>> graph = embedder.embed_s('Thre is a large red rubber ball to the right of a green cylinder')
    """

    _default_backend = 'torch'
    _backend_registry = dict()

    def __init__(self, backend=None, **kwargs):
        _load_backends()

        self.backend = backend
        if self.backend is None:
            self.backend = type(self)._default_backend
        if self.backend not in type(self)._backend_registry:
            raise ValueError('Unknown backend: {}.'.format(self.backend))

        self._init_kwargs = kwargs
        self._inst = type(self)._backend_registry[self.backend](**kwargs)
        #print(f"Instantiated embedder {type(self._inst)} at {hex(id(self._inst))}")

    def __call__(self, *args, **kwargs):
        return self.unwrapped(self)

    @property
    def init_kwargs(self):
        """
        Get the keyward arguments used for initializing the backend.
        """
        return self._init_kwargs

    @property
    def unwrapped(self):
        """
        Get the backend.
        """
        return self._inst

    def embed_s(self, sentence, *args, **kwargs):
        """
        This call parses a sentence into a scene graph using the `parser.parse(..)`
        command. Based on the parser's LM, provides feature embeddings for
        the graph nodes, edge attributes and edge indices.

        These outputs (X, e_i, e_attr) can be readily used for further training using
        any GNN training framework of choice, for e.g. `torch_geometric`.

        For e.g., a ConcreteEmbedder extension can have a function with:
        ```
        data = torch_geometric.Data(x=X, edge_index=e_i, edge_attr=e_attr)
        ```
        Returns a tuple (X, edge_idx, edge_attr)

        Args:
            sentence (str): the input sentence.

        Returns:
            X: the node feature embeddings of shape [V, dim]
            e_i: the edge indices of shape [2, E]
            e_attr: the edge attr feature embeddings of shape [E, dim]

            README file for the specification of the return value.
        """
        return self.unwrapped.embed_s(sentence, *args, **kwargs)

    def embed_t(self, img_idx:int, img_scene_path:str, *args, **kwargs):
        """
        This call parses a image scene graph (obtained by the idx and scene graph path) to
         embeddings (based on the parser's LM), provides feature embeddings for
        the graph nodes, edge attributes and edge indices.

        These outputs (X, e_i, e_attr) can be readily used for further training using
        any GNN training framework of choice, for e.g. `torch_geometric`.

        For e.g., a ConcreteEmbedder extension can have a function with:
        ```
        data = torch_geometric.Data(x=X, edge_index=e_i, edge_attr=e_attr)
        ```
        Returns a tuple (X, edge_idx, edge_attr)

        Args:
            img_idx (int): imgage index
            img_scene_path (str): the path to the image scene graph

        Returns:
            X: the node feature embeddings of shape [V, dim]
            e_i: the edge indices of shape [2, E]
            e_attr: the edge attr feature embeddings of shape [E, dim]

            README file for the specification of the return value.
        """
        return self.unwrapped.embed_t(img_idx, img_scene_path, *args, **kwargs)

    # def get_embeddings(self, G, **kwargs):
    #     """
    #     Parser a Graph into Embeddings
    #     :param G (nx.Graph): the input graph
    #     :param kwargs:
    #     :return: Embeddings
    #     """
    #     return self.unwrapped.get_embeddings(G, **kwargs)

    @classmethod
    def register_backend(cls, backend):
        """
        Register a class as the backend. The backend should implement a
        method named `embed_s` having the following signature:
        `embed_s(sentence, <other_keyward_arguments>)`.

        To register your customized backend as the embedder, use this class
        method as a decorator on your class.

        Example::
        >>> @Embedder.register_backend
        >>> class CustomizedBackend(Backend):
        >>>     # Your implementation follows...
        >>>     pass
        """
        try:
            cls._backend_registry[backend.__identifier__] = backend
        except Exception as e:
            raise ImportError('Unable to register backend: {}.'.format(backend.__name__)) from e

    def get_backend(self, identifier=None, **kwargs):
        if identifier is None:
            return self._default_backend
        if identifier in self._backend_registry:
            _backend = self._backend_registry[identifier]
            if _backend == type(self._inst):
                # Already instantiated
                return self._inst
            return _backend(**kwargs)


# --------------------------------------------------------------------------#
_default_embedder = None

def get_default_embedder():
    """
    Get the default embedder.

    The default embedder is a global one (singleton).
    """
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder

def embed_s(sentence, *args, **kwargs):
    """
    Parse the sentence using the default embedder. This ia an easy-to-use
    feature for those who do not want to configure their own embedders
    and want to use the embedder at different places in their codes.

    Please note that the default embedder is a singleton. Thus,
    if you are using a stateful embedder, you need to be careful about sharing
    this embedder everywhere.
    """
    return get_default_embedder().embed_s(sentence, *args, **kwargs)

def embed_t(img_idx, img_scene_path, *args, **kwargs):
    """
    See @embed_s for ref.
    """
    return get_default_embedder().embed_t(img_idx,img_scene_path, *args, **kwargs)

# def get_embeddings(G, *args, **kwargs):
#     """
#     Parse the Graph using the default embedder. This ia an easy-to-use
#     feature for those who do not want to configure their own embedders
#     and want to use the embedder at different places in their codes.
#
#     Please note that the default embedder is a singleton. Thus,
#     if you are using a stateful embedder, you need to be careful about sharing
#     this embedder everywhere.
#     """
#     return get_default_embedder().get_embeddings(G, *args, **kwargs)

