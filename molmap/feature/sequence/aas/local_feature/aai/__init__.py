#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul  1 13:49:20 2021

@author: wanxiang.shen@u.nus.edu
"""
import os
import pandas as pd


file_path = os.path.dirname(__file__)


class load_all:
    @property
    def data(self):
        return pd.read_csv(os.path.join(file_path, 'result_data', 'all_data.csv'))
    @property
    def meta(self):
        return pd.read_csv(os.path.join(file_path, 'result_data', 'all_meta.csv'))



class load_index:
    @property
    def data(self):
        return pd.read_csv(os.path.join(file_path, 'result_data', '01_aaindex.csv'), index_col = 0)
    @property
    def meta(self):
        return pd.read_pickle(os.path.join(file_path, 'result_data', '01_aaindex_meta.pkl'))

