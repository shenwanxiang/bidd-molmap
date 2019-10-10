#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: charleshen

@note: calculation of AdjacencyMatrix, BaryszMatrix, DetourMatrix, DistanceMatrix descriptors
"""


from mordred import Calculator, descriptors
import numpy as np

_calc = Calculator([descriptors.AdjacencyMatrix,
                     descriptors.BaryszMatrix,
                     descriptors.DetourMatrix,
                     descriptors.DistanceMatrix])


_MatrixNames = [str(i) for i in _calc.descriptors]



def GetMatrix(mol):
    r = _calc(mol)
    r = r.fill_missing(0)
    return r.asdict()