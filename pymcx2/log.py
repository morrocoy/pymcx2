# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Tue Mar 16 21:43:18 2021

@author: kpapke
"""
import logging


class LogManager(object):
    """ Class to manage multiple loggers from different modules
    """

    def __init__(self):
        """ Constructor. """
        self.level = logging.WARNING

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(self.level)

        self.formatter = logging.Formatter(
            "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
            "%(levelname)-7s: %(message)s")
        self.console_handler.setFormatter(self.formatter)

        self.loggers = {}

    def getLogger(self, module_name):
        """ Return a logger for a module.
        """
        log = logging.getLogger(module_name)
        log.setLevel(self.level)
        log.addHandler(self.console_handler)
        self.loggers[module_name] = log
        return log

    def setLevel(self, level):
        """ Set the log level for all loggers that have been created.
        """
        self.level = level
        self.console_handler.setLevel(level)
        for log in self.loggers.values():
            log.setLevel(level)


logmanager = LogManager()
