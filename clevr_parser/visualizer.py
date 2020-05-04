#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: visualizer.py
# Author: Raeid Saqur
# Email: rsaqur@cs.princeton.edu
# Created on: 2020-05-02
# 
# This file is part of CLEVR-PARSER
# Distributed under terms of the MIT License

__all__ = ['Visualizer', 'get_default_visualizer', 'draw_graph']

def _load_backends():
    from . import backends

class Visualizer(object):
    """
    Example::
    >>> visualizer = Visualzier(backend, **init_kwargs)
    >>> graph = visualzier.draw_graph(Gs)
    """
    _default_backend = 'matplotlib'
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
        #print(f"Instantiated visualizer {type(self._inst)} at {hex(id(self._inst))}")

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

    @classmethod
    def draw_graph(cls, G, *args, **kwargs):
        """
        Draw a Graph from text or image scene
        :param G (nx.Graph): the input graph
        :param kwargs:
        :return: Embeddings
        """
        return cls.unwrapped.draw_graph(G, *args, **kwargs)

    @classmethod
    def register_backend(cls, backend):
        """
        Register a class as the backend. The backend should implement a
        method named `draw_graph` having the following signature:
        `draw_graph(G, <other_args>, <other_keyword_args>)`.

        To register your customized backend as the visualizer, use this class
        method as a decorator on your class.

        Example::
        >>> @Visualizer.register_backend
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
_default_visualizer = None

def get_default_visualizer():
    """
    Get the default visualizer (Global singleton)
    """
    global _default_visualizer
    if _default_visualizer is None:
        _default_visualizer = Visualizer()
    return _default_visualizer

def draw_graph(G, *args, **kwargs):
    """
    Please note that the default visualizer is a singleton. Thus,
    if you are using a stateful visualizer, you need to be careful about sharing
    this visualizer everywhere.
    """
    return get_default_visualizer().draw_graph(G, *args, **kwargs)

