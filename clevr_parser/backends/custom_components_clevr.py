#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Raeid Saqur
Email  : rsaqur@cs.princeton.edu
Date   : 12/16/2019

This file is part of PGFM Parser.
Distributed under terms of the MIT license.
https://github.com/raeidsaqur/clevr-parser

* Custom pipeline components: https://spacy.io//usage/processing-pipelines#custom-components
Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0

"""

from __future__ import unicode_literals, print_function

import re
from itertools import permutations
from typing import List, Dict

from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler, EntityLinker, EntityRecognizer
from spacy.tokens import Doc, Span, Token

from clevr_parser.utils import *

__all__ = ['CLEVRObjectRecognizer']

"""
ToDos:
 - Matcher should be used (Regex) for spatial phrases
 - Add Logging, remove print statements
"""

## CLEVR ATTR Pattern Rules ##
C = {"_": {"is_color": {"==": 1}}}
Z = {"_": {"is_size": {"==": 1}}}
M = {"_": {"is_material": {"==": 1}}}
S = {"_": {"is_shape": {"==": 1}}}

attrs = [C, Z, M]
fn = lambda p: list(map(lambda x: x + [S], list(map(lambda y: list(y), permutations(attrs, p)))))
qn = lambda r: list(map(fn, r))


class CLEVRObjectRecognizer(object):
	"""
	spaCy v2.0 pipeline component that sets entity annotations and labels for 
	CLEVR objects
	"""

	name = "clevr_object_recognizer"

	fn = lambda p: list(map(lambda x: x + ['S'], list(map(lambda y: list(y), permutations(attrs, p)))))
	qn = lambda r: list(map(fn, r))

	def __init__(self, nlp, label="CLEVR_OBJ", include_plurals=True):
		
		self.label = nlp.vocab.strings[label]  # get entity label ID
		self.include_plurals = include_plurals
		self._init_constants()
		self._add_custom_spacy_extensions()
		patterns = self.construct_patterns()
		self.ruler = EntityRuler(nlp, phrase_matcher_attr=None, overwrite_ents=False, validate=True)
		self.ruler.add_patterns(patterns)					# label: CLEVR_OBJ
		plural_patterns = self.construct_plural_patterns()  # label: CLEVR_OBJS
		if self.include_plurals:
			self.ruler.add_patterns(plural_patterns)

		self._add_ruler_to_pipeline(nlp, self.ruler, force=True)

	@staticmethod
	def is_equal_size(attr1:str, attr2:str) -> bool:
		if isinstance(attr1, Token):
			attr1 = attr1.text
		if isinstance(attr2, Token):
			attr2 = attr2.text
		_is_eq = False
		if attr1 == attr2:
			_is_eq = True
		else:
			if attr1 in ['small', 'tiny'] and attr2 in ['tiny', 'small']:
				_is_eq = True
			elif attr1 in ['large', 'big'] and attr2 in ['big', 'large']:
				_is_eq = True
		return _is_eq

	@staticmethod
	def is_equal_material(attr1: str, attr2: str) -> bool:
		if isinstance(attr1, Token):
			attr1 = attr1.text
		if isinstance(attr2, Token):
			attr2 = attr2.text
		if attr1 == attr2:
			return True
		cat1_synonyms = ['rubber', 'matte']
		cat2_synonyms = ['metal', 'metallic', 'shiny']
		if attr1 in ['rubber', 'matte'] and attr2 in ['rubber', 'matte']:
			return True
		elif attr1 in cat2_synonyms and attr2 in cat2_synonyms:
			return True
		else:
			return False

	@staticmethod
	def is_equal_shape(attr1: str, attr2: str) -> bool:
		if isinstance(attr1, Token):
			attr1 = attr1.text
		if isinstance(attr2, Token):
			attr2 = attr2.text
		if attr1 == attr2:
			return True
		cat1_synonyms = ['cube', 'block', 'thing', 'object']
		cat2_synonyms = ['sphere', 'ball', 'thing', 'object']
		cat3_synonyms = ['cylinder', 'thing', 'object']
		if attr1 in cat1_synonyms and attr2 in cat1_synonyms:
			return True
		elif attr1 in cat2_synonyms and attr2 in cat2_synonyms:
			return True
		elif attr1 in cat3_synonyms and attr2 in cat3_synonyms:
			return True
		else:
			return False

	def _init_constants(self):
		## Attrs with synonyms.json mixin: ##
		color_attrs = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
		material_attrs = ['rubber', 'matte', 'metal', 'metallic', 'shiny']
		size_attrs = ['small', 'tiny', 'large', 'big']
		shape_attrs = ['cube', 'block', 'sphere', 'ball', 'cylinder', 'thing', 'object']

		is_size_getter = lambda token: token.text in size_attrs
		is_color_getter = lambda token: token.text in color_attrs
		is_material_getter = lambda token: token.text in material_attrs
		is_shape_getter = lambda token: token.text in shape_attrs

		has_size_getter = lambda obj: any([t.text in size_attrs for t in obj])
		has_color_getter = lambda obj: any([t.text in color_attrs for t in obj])
		has_material_getter = lambda obj: any([t.text in material_attrs for t in obj])
		has_shape_getter = lambda obj: any([t.text in shape_attrs for t in obj])

		#size_getter = lambda obj: filter(lambda token: token.text if is_size_getter(token) else '', obj)
		size_getter = lambda obj: list(filter(lambda token: token.text if is_size_getter(token) else '', obj))[0]
		color_getter = lambda obj: list(filter(lambda token: token.text if is_color_getter(token) else '', obj))[0]
		material_getter = lambda obj: list(filter(lambda token: token.text if is_material_getter(token) else '', obj))[0]
		shape_getter = lambda obj: list(filter(lambda token: token.text if is_shape_getter(token) else '', obj))[0]

		is_attrs = ['is_size', 'is_color', 'is_material', 'is_shape']
		has_attrs = ['has_size', 'has_color', 'has_material', 'has_shape']
		get_attrs = ['size', 'color', 'material', 'shape']

		is_attrs_getters = [is_size_getter, is_color_getter, is_material_getter, is_shape_getter]
		has_attrs_getters = [has_size_getter, has_color_getter, has_material_getter, has_shape_getter]
		attrs_getters = [size_getter, color_getter, material_getter, shape_getter]

		if self.include_plurals:
			shapes_attrs = list(map(lambda x: x + 's', shape_attrs))
			is_shapes_getter = lambda token: token.text in shapes_attrs  # Handle plurals
			has_shapes_getter = lambda obj: any([t.text in shapes_attrs for t in obj])  #Handle plurals
			is_attrs = ['is_size', 'is_color', 'is_material', 'is_shape', 'is_shapes']
			has_attrs = ['has_size', 'has_color', 'has_material', 'has_shape', 'has_shapes']
			is_attrs_getters = [is_size_getter, is_color_getter, is_material_getter, is_shape_getter, is_shapes_getter]
			has_attrs_getters = [has_size_getter, has_color_getter, has_material_getter, has_shape_getter,
								 has_shapes_getter]

		assert len(is_attrs) == len(is_attrs_getters)
		assert len(has_attrs) == len(has_attrs_getters)
		assert len(get_attrs) == len(attrs_getters)

		self.is_attrs_name2func = list(zip(is_attrs, is_attrs_getters))
		self.has_attrs_name2func = list(zip(has_attrs, has_attrs_getters))
		self.get_attrs_name2func = list(zip(get_attrs, attrs_getters))

	def __call__(self, doc):
		""" """
		return doc

	def add_event_ent(self, matcher, doc, i, matches):
		#print(f"{__name__}.add_event_ent")
		# Get the current match and create tuple of entity label, start and end.
		# Append entity to the doc's entity. (Don't overwrite doc.ents!)
		match_id, start, end = matches[i]
		entity = Span(doc, start, end, label="EVENT")
		# doc.ents += (entity,)
		#print(f"Matched entity: {entity.text}")

    ############ Attr getter functions #######################
	def _add_ruler_to_pipeline(self, nlp, ruler, adj_comp="ner", after=False, force=False):
		if nlp == None:
			raise ValueError("nlp can NOT be None")
		pipeline, _ = zip(*nlp.pipeline)
		if ruler.name not in pipeline:
			#print(f"Adding {ruler.name} at {hex(id(ruler))} in pipeline at {hex(id(nlp))}")
			if after:
				nlp.add_pipe(ruler, after=adj_comp)
			else:
				nlp.add_pipe(ruler, before=adj_comp)
		else:
			#print(f"{ruler.name} already exists in pipeline\n\t{pipeline}")
			if force:
				#print(f"Force=True, removing and re-adding ruler")
				pass
		pipeline, _ = zip(*nlp.pipeline)
		#print(f"Updated pipeline: {pipeline}")

	def _remove_ruler_from_pipeline(self, nlp, ruler, adj_comp="ner", after=False):
		if nlp == None:
			raise ValueError("nlp can NOT be None")
		pipeline, _ = zip(*nlp.pipeline)
		if ruler.name not in pipeline:
			print(f"'entity_ruler' not in pipeline:\n\t{pipeline}")
		nlp.remove_pipe(ruler.name)
		pipeline, _ = zip(*nlp.pipeline)
		#print(f"Removed {ruler.name}, updated pipeline: {pipeline}")

	def _add_custom_spacy_extensions(self):
		for n, f in self.is_attrs_name2func:
			ext = Token.get_extension(n)
			if ext is None:
				Token.set_extension(n, getter=f, force=True)
		for item in [Span, Doc]:
			for n, f in self.has_attrs_name2func:
				ext = item.get_extension(n)
				if ext is None:
					#print(f"Setting: {item}.set_extension({n}, getter= {f})")
					item.set_extension(n, getter=f, force=True)

		# Add Attr Getters for Span (i.e. Doc.ents)
		for n, f in self.get_attrs_name2func:
			ext = Span.get_extension(n)
			if ext is None:
				Span.set_extension(n, getter=f, force=True)


	@staticmethod
	def construct_patterns(label="CLEVR_OBJ") -> List[Dict]:
		def _patterns() -> List[List[List]]:
			C = {"_": {"is_color": {"==": 1}}}
			Z = {"_": {"is_size": {"==": 1}}}
			M = {"_": {"is_material": {"==": 1}}}
			S = {"_": {"is_shape": {"==": 1}}}
			attrs = [C, Z, M]

			fn = lambda p: list(map(lambda x: x + [S], list(map(lambda y: list(y), permutations(attrs, p)))))
			qn = lambda r: list(map(fn, r))
			return qn([3, 2, 1, 0])

		obj_patterns = []
		for patterns in _patterns():
			for pattern in patterns:
				obj_pattern = {"label": label, "pattern": pattern}
				obj_patterns.append(obj_pattern)

		return obj_patterns

	@staticmethod
	def construct_plural_patterns(label="CLEVR_OBJS") -> List[Dict]:
		"""
		E.g. case (from CLEVR q.fam: 2): 'Are there more big green things than large purple shiny cubes?'
		:param label: entity label
		:return: pattern
		"""
		def _patterns() -> List[List[List]]:
			C = {"_": {"is_color": {"==": 1}}}
			Z = {"_": {"is_size": {"==": 1}}}
			M = {"_": {"is_material": {"==": 1}}}
			SS = {"_": {"is_shapes": {"==": 1}}}
			attrs = [C, Z, M]

			fn = lambda p: list(map(lambda x: x + [SS], list(map(lambda y: list(y), permutations(attrs, p)))))
			qn = lambda r: list(map(fn, r))
			return qn([3, 2, 1, 0])

		obj_patterns = []
		for patterns in _patterns():
			for pattern in patterns:
				obj_pattern = {"label": label, "pattern": pattern}
				obj_patterns.append(obj_pattern)

		return obj_patterns


	@staticmethod
	def _get_token_info(doc, is_debug=False):
		if is_debug:
			for token in doc:
				print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}".format(
					token.text,
					token.idx,
					token.lemma_,
					token.is_punct,
					token.is_space,
					token.shape_,
					token.pos_,
					token.tag_,
					token._.is_color
				))
