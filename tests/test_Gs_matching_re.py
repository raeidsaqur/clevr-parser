#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: test_Gs_matching_re.py
# Author: raeidsaqur
# Email: rsaqur@cs.princeton.edu
# Created on: 2020-05-03
# 
# This file is part of CLEVR Parser
# Distributed under terms of the MIT License

import pytest
import clevr_parser
from clevr_parser.utils import *

# Test Fixtures #
@pytest.fixture(scope="module")
def parser():
    parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                 has_spatial=True,
                                 has_matching=True).get_backend(identifier='spacy')
    return parser

def test_Gs_matching_relation(parser):
    text = "Does the sphere behind the rubber cylinder have the same color as the metal cube"
    Gs, s_doc = parser.parse(text, return_doc=True)
    matching_res = parser.filter_matching_re(s_doc.ents)
    assert matching_res is not None
    assert len(matching_res) == 1

