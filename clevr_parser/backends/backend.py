#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : backend.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/21/2019
#
# This file is part of PGFM Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

from abc import ABC, abstractmethod, abstractproperty

__all__ = ['ParserBackend', 'EmbedderBackend', 'VisualizerBackend']

class ParserBackend(object):
    """
    Based class for all parser backends. This class
    specifies the methods that should be override by subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def parse(self, sentence):
        raise NotImplementedError()

class EmbedderBackend(object):
    """
    Based class for all embedder backends. This class
    specifies the methods that should be override by subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        #super(EmbedderBackend, self).__init__()

    def embed(self, sentence):
        raise NotImplementedError()

    def get_embeddings(self, G):
        raise NotImplementedError()

class VisualizerBackend(ABC):
    """
    Based class for all visualizer backends. This class
    specifies the methods that should be override by subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    @abstractmethod
    def draw_graph(cls, G, *args, **kwargs):
        raise NotImplementedError()
