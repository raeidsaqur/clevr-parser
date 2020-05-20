#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: parser.py
# Author: Raeid Saqur
# Email: rsaqur@cs.princeton.edu
# Created on: 2020-05-02
#
# This file is part of CLEVR-PARSER
# Distributed under terms of the MIT License

from .utils import trace

__all__ = ['Parser', 'get_default_parser', 'parse']

def _load_backends():
    from . import backends


class Parser(object):
    """
    Example::
    >>> parser = Parser(backend, **init_kwargs)
    >>> graph = parser.parse('Thre is a large red rubber ball to the right of a green cylinder')
    """

    _default_backend = 'spacy'
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
        #print(f"Instantiated parser {type(self._inst)} at {hex(id(self._inst))}")

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

    def parse(self, sentence, *args, **kwargs):
        """
        Parse a sentence into a scene graph.

        Args:
            sentence (str): the input sentence.

        Returns:
            graph (dict): the parsed scene graph. Please refer to the
            README file for the specification of the return value.
        """
        return self.unwrapped.parse(sentence, *args, **kwargs)

    @classmethod
    def register_backend(cls, backend):
        """
        Register a class as the backend. The backend should implement a
        method named `parse` having the following signature:
        `parse(sentence, <other_keyward_arguments>)`.

        To register your customized backend as the parser, use this class
        method as a decorator on your class.

        Example::
        >>> @Parser.register_backend
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
_default_parser = None

def get_default_parser():
    """
    Get the default parser.

    The default parser is a global one (singleton).
    """
    global _default_parser
    if _default_parser is None:
        _default_parser = Parser()
    return _default_parser


def parse(sentence, *args, **kwargs):
    """
    Parse the sentence using the default parser. This ia an easy-to-use
    feature for those who do not want to configure their own parsers
    and want to use the parser at different places in their codes.

    Please note that the default parser is a singleton. Thus,
    if you are using a stateful parser, you need to be careful about sharing
    this parser everywhere.
    """
    return get_default_parser().parse(sentence, *args, **kwargs)

