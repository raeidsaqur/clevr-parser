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

__all__ = ['StanfordnlpParser']

from typing import Dict, Tuple, Sequence
import logging
logger = logging.getLogger(__name__)


@Parser.register_backend
class StanfordnlpParser(ParserBackend):
    """
    Scene graph parser based on spaCy.
    """

    __identifier__ = 'stanfordnlp'

    def __init__(self, model='en'):
        """
        Args:
            model (str): a spec for the spaCy model. (default: en). Please refer to the
            official website of spaCy for a complete list of the available models.
            This option is useful if you are dealing with languages other than English.
        """

        self.model = model

        try:
            MODELS_DIR = '~/stanfordnlp_resources'
            import stanfordnlp
            stanfordnlp.download(model, treebank="en_ewt", resource_dir=MODELS_DIR, confirm_if_exists=True)
        except ImportError as e:
            raise ImportError('StanfordNLP backend requires the stanfordnlp library. Install spaCy via pip install stanfordnlp.') from e

        try:
            config = {
                'processors': 'tokenize,pos,lemma,depparse',
                'tokenize_pretokenized': True,
                'models_dir': f'{MODELS_DIR}',
                'treebank': 'en_ewt',
                'pos_model_path': f'{MODELS_DIR}/en_ewt_models/en_ewt_tagger.pt',
                'pos_pretrain_path': f'{MODELS_DIR}/en_ewt_models/en_ewt.pretrain.pt',
                'pos_batch_size': 1000
            }
            self.nlp = stanfordnlp.Pipeline(**config)
        except OSError as e:
            raise ImportError('Unable to load the English model. Run `stanfordnlp.download(model, MODELS_DIR)` first.') from e

    def parse(self, sentence:str, return_doc=False):
        """
                The spaCy-based parser parse the sentence into scene graphs based on the dependency parsing
                of the sentence by spaCy.

                All entities (nodes) of the graph come from the noun chunks in the sentence. And the dependencies
                between noun chunks are used for determining the relations among these entities.

                The parsing is performed in three steps:

                    1. find all the noun chunks as the entities, and resolve the modifiers on them.
                    2. determine the subject of verbs (including nsubj, acl and pobjpass). Please refer to the comments
                    in the code for better explanation.
                    3. determine all the relations among entities.
                """
        print(f"stanfordnlp_parser.parse(sentence: {sentence}")
        doc = self.nlp(sentence)
        doc.sentences[0].print_tokens()  # Look at the result
        print(*[f'text: {word.text + " "}\tlemma: {word.lemma}\tupos: {word.upos}\txpos: {word.xpos}' for sent in
                doc.sentences for word in sent.words], sep='\n')

        # Step 1: determine the entities.
        entities = list()
        entity_chunks = list()

        # TODO: Incomplete the parsing logic for stanfordnlp parser wil be
        # different than spacy. Complete. See spacy_parser.py

        relations = list()
        filtered_relations = list()


        if return_doc:
            return {'entities': entities, 'relations': filtered_relations}, doc
        return {'entities': entities, 'relations': filtered_relations}


    @staticmethod
    def __locate_noun(chunks, i):
        for j, c in enumerate(chunks):
            if c.start <= i < c.end:
                return j
        return None

