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

__all__ = ['ParserBackend']


class ParserBackend(object):
    """
    Based class for all parser backends. This class
    specifies the methods that should be override by subclasses.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        #super(ParserBackend, self).__init__()

    def parse(self, sentence):
        raise NotImplementedError()

