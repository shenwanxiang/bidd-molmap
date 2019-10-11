#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:54:12 2019

@author: wanxiang.shen@u.nus.edu

@logtools
"""


import os, sys, logging, time, traceback, inspect
from colorlog import ColoredFormatter
from colored import fg, bg, attr

formatter = ColoredFormatter(
    "%(asctime)s - %(log_color)s%(levelname).4s%(reset)s - %(message_log_color)s[%(name)s]%(reset)s - %(message_log_color)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={
        'message':{
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
    },
    style='%',
    )


logger = logging.getLogger('bidd-molmap')
logger.propagate = False


all_loggers = [ logger]

def set_level(level):
    for _logger in all_loggers:
        _logger.setLevel(getattr(logging, level.upper()))

set_level('INFO')

file_handler = None
def log_to_file(path):
    global file_handler
    if file_handler is not None:
        for _logger in all_loggers:
            _logger.removeHandler(file_handler)
            
    logpath = os.path.join(os.getcwd(), path + '.' + get_datetime() + '.log')
    print_info('log to file:', logpath)
    file_handler = logging.FileHandler(logpath)
    file_handler.setFormatter(formatter)
    for _logger in all_loggers:
        _logger.addHandler(file_handler)

def reset_handler(handler):
    for _logger in all_loggers:
        del _logger.handlers[:]
        _logger.addHandler(handler)
        if file_handler is not None:
            _logger.addHandler(file_handler)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
reset_handler(handler)

DEFAULT_TEXT_LENGTH = 5000
DEFAULT_TEXT_LENGTH_PREFIX = 4000
DEFAULT_TEXT_LENGTH_SUFFIX = DEFAULT_TEXT_LENGTH - DEFAULT_TEXT_LENGTH_PREFIX
def set_text_length(prefix, suffix):
    global DEFAULT_TEXT_LENGTH, DEFAULT_TEXT_LENGTH_PREFIX, DEFAULT_TEXT_LENGTH_SUFFIX
    DEFAULT_TEXT_LENGTH = prefix + suffix
    DEFAULT_TEXT_LENGTH_PREFIX = prefix
    DEFAULT_TEXT_LENGTH_SUFFIX = suffix

def clip_text(text):
    if len(text) > DEFAULT_TEXT_LENGTH:
        text = '%s %s%s... [%d chars truncated] ...%s %s' % (text[:DEFAULT_TEXT_LENGTH_PREFIX], fg('red'), attr('bold'), len(text) - DEFAULT_TEXT_LENGTH, attr('reset'), text[-DEFAULT_TEXT_LENGTH_SUFFIX:])
    return text

def create_print_method(level):
    print_method = getattr(logger, level)
    def func(*args, sep=' ', verbose=True):
        if verbose: print_method(clip_text(sep.join(map(str, args))))
    return func

print_error = create_print_method('error')
print_warn = create_print_method('warn')
print_info = create_print_method('info')
print_debug = create_print_method('debug')

def format_exc(error):
    return traceback.format_exception(error.__class__, error, error.__traceback__)

def print_exc(error, verbose=True):
    lines = format_exc(error)
    logger.error(lines[-1].rstrip())
    if verbose:
        logger.info(''.join(lines[:-1]))

def print_exc_s(error):
    logger.error('%s: %s', type(error).__name__, error)

def print_traceback():
    logger.info('Traceback (most recent call last):')
    for frame in reversed(inspect.getouterframes(inspect.currentframe())[1:]):
        logger.info('  File "%s", line %s, in %s', frame.filename, frame.lineno, frame.function)
        logger.info('    %s', frame.code_context[0].strip())

last_time = time.time()
def print_timedelta(*args, sep=' '):
    global last_time
    this_time = time.time()
    logger.info('[%7.2f] %s', (this_time - last_time) * 1000, sep.join(map(str, args)))
    last_time = this_time

def get_date():
    return time.strftime('%Y%m%d')

def get_datetime():
    return time.strftime('%Y%m%d%H%M%S')

from tqdm import tqdm

class PBarHandler(logging.Handler):

    def __init__(self, pbar):
        logging.Handler.__init__(self)
        self.pbar = pbar

    def emit(self, record):
        self.pbar.write(self.format(record))

def pbar(*args, **kwargs):
    kwargs['ascii'] = kwargs.get('ascii', True)
    kwargs['smoothing'] = kwargs.get('smoothing', 0.7)

    pb = tqdm(*args, **kwargs)
    handler = PBarHandler(pb)
    handler.setFormatter(formatter)

    pb.handler = handler
    reset_handler(handler)

    return pb

if __name__ == '__main__':
    pb = pbar(range(100))
    for i in pb:
        time.sleep(0.05)
        print_info(str(i))
