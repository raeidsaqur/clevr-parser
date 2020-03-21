#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : database.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/23/2019
#
# This file is part of PGFM Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

import os
import json


_caches = dict()

def load_list(filename):
    if filename not in _caches:
        out = set()
        for x in open(os.path.join(os.path.dirname(__file__), '_data', filename)):
            x = x.strip()
            if len(x) > 0:
                out.add(x)
        _caches[filename] = out
    return _caches[filename]


def is_phrasal_verb(verb):
    return verb in load_list('phrasal-verbs.txt')


def is_phrasal_prep(prep):
    return prep in load_list('phrasal-preps.txt')

## Attributes ##
def is_spatial_prep(prep):
    print(f"is_spatial_prep: {prep}")
    return prep in load_list('spatial-preps.txt')

def is_attribute_relation(prep):
    return prep in load_list('relation-attrs.txt')


def is_attribute_color(prep):
    return prep in load_list('color-attrs.txt')

# TODO: should also load and check for synonyms (from synonyms.json)
def is_attribute_size(prep):
    return prep in load_list('size-attrs.txt')

def is_attribute_material(prep):
    return prep in load_list('material-attrs.txt')

def is_attribute_shape(prep):
    return prep in load_list('shape-attrs.txt')



def is_scene_noun(noun):
    head = noun.split(' ')[-1]
    #TODO:RS (lift logic)
    # s = load_list('clevr-scene-nouns.txt')
    s = load_list('scene-nouns.txt')
    return noun in s or head in s

