#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File   : spatial_recognizer.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/21/2019
#
# This file is part of CLEVR Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

from __future__ import unicode_literals, print_function

import re
from itertools import permutations
from typing import List, Dict

from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler, EntityLinker, EntityRecognizer
from spacy.tokens import Doc, Span, Token

from clevr_parser.utils import *
from clevr_parser import setup_logging
import logging
logger = setup_logging(__name__, log_level=logging.DEBUG)

__all__ = ['SpatialRecognizer']

class SpatialRecognizer(object):

    name = "spatial_recognizer"

    def __init__(self, nlp, label="SPATIAL_RE", **kwargs):
        self.label = nlp.vocab.strings[label]
        self.nlp = nlp
        self.ruler = EntityRuler(nlp, overwrite_ents=False, validate=True)
        self.is_debug = kwargs.get("is_debug")
        try:
            self.ruler = self.ruler.from_disk("../_data/spatial-re-patterns.jsonl")
            self._add_ruler_to_pipeline(nlp, self.ruler)
        except ValueError as ve:
            if self.is_debug:
                logger.error(f"{ve}: Ensure patterns file is added.")
            self._add_patterns()
        if self.is_debug:
            logger.debug(f"Pipeline -> {nlp.pipe_names}")

    def __call__(self, doc):
        return doc

    def construct_patters(self):
        """Load patterns adhoc if loading .jsonl file fails"""
        obj_patterns = [
            {"label": "SPATIAL_RE",
             "pattern": [{"LOWER": "that"}, {"TEXT": {"IN": ["is", "are"]}}, {"LOWER": "behind"}, {"LOWER": "the"}]},
            {"label": "SPATIAL_RE", "pattern": [{"LOWER": "behind"}, {"LOWER": "the"}]},
            {"label": "SPATIAL_RE",
             "pattern": [{"LOWER": "that"}, {"TEXT": {"IN": ["is", "are"]}}, {"LOWER": "in"}, {"LOWER": "front"}]},
            {"label": "SPATIAL_RE",
             "pattern": [{"LOWER": "that"}, {"TEXT": {"IN": ["is", "are"]}}, {"TEXT": {"IN": ["in", "on"]}},
                         {"LOWER": "the"}, {"TEXT": {"IN": ["left", "right"]}}]},
            {"label": "SPATIAL_RE", "pattern": [{"LOWER": "to"}, {"LOWER": "the"}, {"TEXT": {"IN": ["right", "left"]}},
                                                {"TEXT": {"IN": ["", "of"]}}]},
            {"label": "SPATIAL_RE", "pattern": [{"TEXT": {"IN": ["above", "below"]}}, {"LOWER": "the"}]},
            {"label": "SPATIAL_RE", "pattern": [{"TEXT": {"IN": ["left", "right", "front", "behind", "above", "below"] }}, ]}
        ]

        return obj_patterns

    def _add_patterns(self):
        patterns = self.construct_patters()
        #other_pipes = [p for p in self.nlp.pipe_names if p != "tagger"] # excluse tagger
        other_pipes = self.nlp.pipe_names
        with self.nlp.disable_pipes(*other_pipes):
            self.ruler.add_patterns(patterns)
        #self.ruler.add_patterns(patterns)
        self._add_ruler_to_pipeline(self.nlp, self.ruler)

    def _add_ruler_to_pipeline(self, nlp, ruler, name="spatial_entity_ruler"):
        if nlp.has_pipe(name):
            nlp.replace_pipe(name, self.ruler)
        else:
            nlp.add_pipe(self.ruler, name=name, last=True)

