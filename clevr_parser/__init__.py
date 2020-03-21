#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Raeid Saqur
# Email  : raeidsaqur@gmail.com
# Date   : 09/21/2019
#
# This file is part of PGFM Parser.
# Distributed under terms of the MIT license.
# https://github.com/raeidsaqur/clevr-parser

from ._version import __version__, __author__, __email__

from .parser import *
from .utils import *
from .explacy import *


import os, logging

def setup_logging(name, log_level=logging.INFO, log_file=None):
    """
    E.g. setup, in '___main__'
    log_fn = f"process_clevr_data.py_{args.dataset}_{args.split}.out"
    logger = setup_logging(__name__, log_file=log_fn)
    logger.info(f'Called with args: {args}')

    main(args)

    :param name:
    :param log_level:
    :param log_file:
    :return:
    """
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    #logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logging.basicConfig(level=log_level, format=FORMAT)
    logger = logging.getLogger(name)
    if log_file:
        log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_fp = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(log_fp, encoding="utf-8", mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(console_handler)

    logger.setLevel(log_level)
    return logger


