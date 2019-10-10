#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: charleshen

316 Estate Index descriptors
"""


from mordred import Calculator, descriptors
import numpy as np

_calc = Calculator(descriptors.EState)

_EstateNames = [str(i) for i in _calc.descriptors]


def GetEstate(mol):
    r = _calc(mol)
    r = r.fill_missing(0)
    return r.asdict()

