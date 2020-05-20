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
from typing import *

__all__ = ['ParserBackend', 'EmbedderBackend', 'VisualizerBackend', 'EmbeddingVisualizerBackend']

class ParserBackend(object):
    """
    Base class for all parser backends. This class
    specifies the methods that should be overriden by subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def parse(self, sentence, *args, **kwargs):
        raise NotImplementedError()

class EmbedderBackend(object):
    """
    Base class for all embedder backends. This class
    specifies the methods that should be overriden by subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        #super(EmbedderBackend, self).__init__()

    @abstractmethod
    def embed_s(self, sentence, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def embed_t(self, img_idx:int, img_scene_path:str, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    # def get_embeddings(self, G, *args, **kwargs):
    #     raise NotImplementedError()

class VisualizerBackend(ABC):
    """
    Base class for all visualizer backends. This class
    specifies the methods that should be overriden by subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    @abstractmethod
    def draw_graph(cls, G, *args, **kwargs):
        raise NotImplementedError()

class EmbeddingVisualizerBackend(ABC):
    """
    Base class for all embedding_visualizer backends. This class
    specifies the methods that should be overriden by subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    @abstractmethod
    def draw_embeddings(cls, vectors, *args, **kwargs):
        raise NotImplementedError()