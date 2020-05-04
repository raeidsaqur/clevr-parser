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

__all__ = ['Embedder', 'get_default_embedder', 'embed', 'get_embeddings']

def _load_backends():
    from . import backends

class Embedder(object):
    """
    Example::
    >>> embedder = Embedder(backend, **init_kwargs)
    >>> graph = embedder.embed('A woman is playing the piano,')
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

    def embed(self, sentence, **kwargs):
        """
        Parse a sentence into a scene graph.

        Args:
            sentence (str): the input sentence.

        Returns:
            graph (dict): the embedd scene graph. Please refer to the
            README file for the specification of the return value.
        """
        return self.unwrapped.embed(sentence, **kwargs)

    def get_embeddings(self, G, **kwargs):
        """
        Parser a Graph into Embeddings
        :param G (nx.Graph): the input graph
        :param kwargs:
        :return: Embeddings
        """
        return self.unwrapped.get_embeddings(G, **kwargs)

    @classmethod
    def register_backend(cls, backend):
        """
        Register a class as the backend. The backend should implement a
        method named `embed` having the following signature:
        `embed(sentence, <other_keyward_arguments>)`.

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


def embed(sentence, **kwargs):
    """
    Parse the sentence using the default embedder. This ia an easy-to-use
    feature for those who do not want to configure their own embedders
    and want to use the embedder at different places in their codes.

    Please note that the default embedder is a singleton. Thus,
    if you are using a stateful embedder, you need to be careful about sharing
    this embedder everywhere.
    """
    return get_default_embedder().embed(sentence, **kwargs)

def get_embeddings(G, **kwargs):
    """
    Parse the Graph using the default embedder. This ia an easy-to-use
    feature for those who do not want to configure their own embedders
    and want to use the embedder at different places in their codes.

    Please note that the default embedder is a singleton. Thus,
    if you are using a stateful embedder, you need to be careful about sharing
    this embedder everywhere.
    """
    return get_default_embedder().get_embeddings(G, **kwargs)

