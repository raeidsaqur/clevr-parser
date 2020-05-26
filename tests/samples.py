#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File: samples.py
# Author: raeidsaqur
# Email: rsaqur@cs.princeton.edu
# Created on: 2020-04-16
# 
# This file is part of Clevr-Parser
# Distributed under terms of the MIT License

import os, sys, platform
import json
import random

import collections; from collections import Counter
from typing import Counter, Tuple, List
from itertools import product

nb_dir = os.getcwd().split()[0]
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)

# rundir = root, tests/samples.py
import clevr_parser
from clevr_parser.explacy import print_parse, print_parse_info
from clevr_parser.utils import *

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")

__all__ = ['TEMPLATES', 'get_s_sample',
           'show_Gs', 'sample_questions_from', 'get_questions_for_img']

TEMPLATES=("and_mat_spa",
              "compare_mat",
              "compare_mat_spa",
              "embed_mat_spa",
              "embed_spa_mat",
              "or_mat",
              "or_mat_spa")

def get_s_sample(template:str, dist='train') -> str:
    """
    :param template: Question family
    :param dist: 'train' or 'test'
    :return: A sample question from template and distribution
    """
    if template not in TEMPLATES:
        raise ValueError("Unknown template type")

    suffix = "baseline" if dist == 'train' else "val"
    template = f"{template}_{suffix}"
    """
    [and_mat_spa_baseline]
    Final program module = query_color
    Question type: query, answer: cyan, question for CLEVR_val_011582.png
    """
    s_ams_bline = "There is a thing that is on the right side of the tiny cyan rubber thing " \
                  "and to the left of the large green matte cylinder; what is its color?"

    """
    [and_mat_spa_val]
    Final program module = query_size
    Question type: query, answer: small, question for CLEVR_val_000019.png, 
    """
    s_ams_val = "What is the size of the thing that is in front of the big yellow object " \
                "and is the same shape as the big green thing?"

    """
    Final program module = count
    Question type: count, answer: 2, question for CLEVR_val_008452.png, 
    """
    s_oms_bline = "How many things are either small green objects in front of the small purple cylinder " \
                  "or large metallic things that are behind the red matte thing ?"

    """
    Final program module = count
    Question type: count, answer: 2, question for CLEVR_val_000439.png, 
    """
    s_oms_val = "How many things are cylinders that are behind the large purple metal thing " \
                "or purple cylinders that are the same size as the cyan thing ?"

    if template == f"and_mat_spa_baseline":
        return s_ams_bline
    elif template == f"and_mat_spa_val":
        return s_ams_val
    elif template == f"or_mat_spa_baseline":
        return s_oms_bline
    elif template == f"or_mat_spa_val":
        return s_oms_val
    else:
        raise ValueError("template must be one of [and|or]_mat_spa_[baseline|val]")
    return None


def get_questions_for_img(img_index, fp, template=None) -> List[str]:
    with open(fp, 'r') as f:
        all_questions = json.load(f)['questions']
        l = len(all_questions)
    if template:
        logging.debug("=" * 50)
        logging.debug(f'{template}')
        logging.debug("-" * 50)
    questions = list(filter(lambda x: x['image_index'] == img_index, all_questions))
    qs = list(map(lambda x: x["question"], questions))
    answers = list(map(lambda x: x['answer'], questions))
    img_filenames = list(map(lambda x: x['image_filename'], questions))
    programs = list(map(lambda x: x['program'], questions))
    items = list(zip(qs, programs, answers, img_filenames))
    visualize_batch_info(items)


def sample_questions_from(fp, template=None, k=10, logging_info=False) -> List[Tuple]:
    with open(fp, 'r') as f:
        all_questions = json.load(f)['questions']
        l = len(all_questions)
        # qtype_counter, q_types = get_qtype_distribution_from_questions(all_questions)
    if template:
        logging.debug("=" * 50)
        logging.debug(f'{template}')
        logging.debug("-" * 50)
    logging.debug(f'Total # of questions = {l}')
    # Sample 10 questions
    q_indices = random.sample(range(l), k=k)
    logging.debug(f'\nq_indices = {q_indices}\n')
    # logging.debug(f'Total # of sampled questions = {len(questions)}')
    questions = []
    for i in q_indices:
        questions.append(all_questions[i])
    # logging.debug(questions)
    qs = list(map(lambda x: x["question"], questions))
    answers = list(map(lambda x: x['answer'], questions))
    img_filenames = list(map(lambda x: x['image_filename'], questions))
    programs = list(map(lambda x: x['program'], questions))
    items = list(zip(qs, programs, answers, img_filenames))
    if logging_info:
        visualize_batch_info(items)
    # population_dist = (qtype_counter, q_types)
    # return population_dist
    return items

def visualize_batch_info(items: List[Tuple]) -> None:
    # similar to dataloder to question_parser
    for q, p, a, img in items:
        _, doc = clevr_parser.parse(q)
        pT = p[-1]['function']
        # q_type = find_clevr_question_type(pT)
        logging.debug('-' * 50)
        logging.debug(f'Final program module = {pT}')
        logging_prog_sequence(p)
        clevr_parser.visualize(doc)
        logging.debug('-' * 50)
        
def logging_prog_sequence(P):
    seq = f"["
    for i, pi in enumerate(P):
        fni = pi['function']
        vis = pi['value_inputs']
        seq += fni+'('
        for vi in vis:
            seq += vi + ', '
        seq += ') '
    seq += " ]"
    logging.debug(seq)
    
def show_Gs(s:str, show_tree=True):
    try:
        Gs, s_doc = clevr_parser.parse(s, return_doc=True)
    except ValueError as ve:
        logging.warning(f"ValueError Encountered: {ve}")
    if Gs is None and ("SKIP" in s_doc):
        logging.warning("Got None as Gs and 'SKIP' in Gs_embd. (likely plural with CLEVR_OBJS label) ")
    logging.debug(f"doc = {s_doc}")
    logging.debug([(ent.text, ent.label_) for ent in s_doc.ents])
    if show_tree:
        print_parse(s_doc)
    clevr_parser.visualize(s_doc, dep=False)