#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of Logger
"""
import io
import logging
from pykg2vec.utils.logger import Logger

def test_logger():
    logger_name = "TEST"
    log = Logger()
    logger = log.get_logger(logger_name)
    captured = io.StringIO()
    test_handler = logging.StreamHandler(captured)
    test_handler.setFormatter(logging.Formatter(Logger.FORMAT))
    logger.addHandler(test_handler)
    message = "This is a log message"

    log.level = logging.WARNING
    logger.debug(message)
    logger.info(message)
    logger.warning(message)
    logger.error(message)
    logger.critical(message)
    another_logger = log.get_logger("ANOTHER")

    result = captured.getvalue()
    captured.close()
    assert not "%s - DEBUG - %s" % (logger_name, message) in result
    assert not "%s - INFO - %s" % (logger_name, message) in result
    assert "%s - WARNING - %s" % (logger_name, message) in result
    assert "%s - ERROR - %s" % (logger_name, message) in result
    assert "%s - CRITICAL - %s" % (logger_name, message) in result
    assert Logger() is log
    assert another_logger.level == log.level
