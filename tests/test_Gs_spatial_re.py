#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_Gs_spatial_re.py
# Author : Raeid Saqur
# Email  : rsaqur@cs.princeton.edu
# Date   : 04/30/2020
#
# This file is part of CLEVR Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

import clevr_parser
import pytest
from clevr_parser.utils import *

@pytest.fixture(scope="module")
def parser():
    parser = clevr_parser.Parser(backend='spacy', model='en_core_web_sm',
                                 has_spatial=True,
                                 has_matching=True).get_backend(identifier='spacy')
    return parser

def test_Gs_spatial_relation(parser):
    text = "The sphere is behind a rubber cylinder right of a metal cube"
    Gs, s_doc = parser.parse(text, return_doc=True)
    spatial_res = parser.filter_spatial_re(s_doc.ents)
    assert spatial_res is not None
    assert len(spatial_res) == 2
