#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul  1 13:49:20 2021

@Orignal author: Stefan Stojanovic; stefs304@gmail.com

@developer: Shen Wan Xiang; wanxiang.shen@u.nus.edu
"""


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


"""
format: https://www.genome.jp/aaindex/aaindex_help.html

Data format of AAindex
Data Format of AAindex1
************************************************************************
*                                                                      *
* Each entry has the following format.                                 *
*                                                                      *
* H Accession number                                                   *
* D Data description                                                   *
* R PMID                                                               *
* A Author(s)                                                          *
* T Title of the article                                               *
* J Journal reference                                                  *
* * Comment or missing                                                 *
* C Accession numbers of similar entries with the correlation          *
*   coefficients of 0.8 (-0.8) or more (less).                         *
*   Notice: The correlation coefficient is calculated with zeros       *
*   filled for missing values.                                         *
* I Amino acid index data in the following order                       *
*   Ala    Arg    Asn    Asp    Cys    Gln    Glu    Gly    His    Ile *
*   Leu    Lys    Met    Phe    Pro    Ser    Thr    Trp    Tyr    Val *
* //                                                                   *
************************************************************************
Data Format of AAindex2 and AAindex3
************************************************************************
*                                                                      *
* Each entry has the following format.                                 *
*                                                                      *
* H Accession number                                                   *
* D Data description                                                   *
* R PMID                                                               *
* A Author(s)                                                          *
* T Title of the article                                               *
* J Journal reference                                                  *
* * Comment or missing                                                 *
* M rows = ARNDCQEGHILKMFPSTWYV, cols = ARNDCQEGHILKMFPSTWYV           *
*   AA                                                                 *
*   AR RR                                                              *
*   AN RN NN                                                           *
*   AD RD ND DD                                                        *
*   AC RC NC DC CC                                                     *
*   AQ RQ NQ DQ CQ QQ                                                  *
*   AE RE NE DE CE QE EE                                               *
*   AG RG NG DG CG QG EG GG                                            *
*   AH RH NH DH CH QH EH GH HH                                         *
*   AI RI NI DI CI QI EI GI HI II                                      *
*   AL RL NL DL CL QL EL GL HL IL LL                                   *
*   AK RK NK DK CK QK EK GK HK IK LK KK                                *
*   AM RM NM DM CM QM EM GM HM IM LM KM MM                             *
*   AF RF NF DF CF QF EF GF HF IF LF KF MF FF                          *
*   AP RP NP DP CP QP EP GP HP IP LP KP MP FP PP                       *
*   AS RS NS DS CS QS ES GS HS IS LS KS MS FS PS SS                    *
*   AT RT NT DT CT QT ET GT HT IT LT KT MT FT PT ST TT                 *
*   AW RW NW DW CW QW EW GW HW IW LW KW MW FW PW SW TW WW              *
*   AY RY NY DY CY QY EY GY HY IY LY KY MY FY PY SY TY WY YY           *
*   AV RV NV DV CV QV EV GV HV IV LV KV MV FV PV SV TV WV YV VV        *
* //                                                                   *
************************************************************************
"""


class Aaindex:

    search_url = 'https://www.genome.jp/dbget-bin/www_bfind_sub?'
    record_url = 'https://www.genome.jp/dbget-bin/www_bget?'
    full_list_url = {
        'aaindex': 'https://www.genome.jp/aaindex/AAindex/list_of_indices',
        'aaindex1': 'https://www.genome.jp/aaindex/AAindex/list_of_indices',
        'aaindex2': 'https://www.genome.jp/aaindex/AAindex/list_of_matrices',
        'aaindex3': 'https://www.genome.jp/aaindex/AAindex/list_of_potentials'
    }

    def __init__(self, source='web'):
        
        '''
        return a pandas dataframe object
        '''
        pass

    def search(self, keyword, dbkey='aaindex', max_hits=0):
        
        '''
        keyword: 'charge', 'alpha-helix' or 'hydrophobicity', ... 
        dbkey: {'aaindex', 'aaindex1','aaindex2', 'aaindex3'}
        '''
        loc = 'locale=en'
        serv = 'serv=gn'
        keywords = 'keywords=' + '+'.join(keyword.split())
        page = 'page=1'
        max_hits = f'max_hit={max_hits}'
        dbkey = f'dbkey={dbkey}'
        params = '&'.join([loc, serv, keywords, page, max_hits, dbkey])
        url = ''.join([self.search_url, params])
        r = requests.get(url)
        if r.status_code == 200:
            return self._parse_search_response(r)

    def get_all(self, dbkey='aaindex'):
        url = self.full_list_url[dbkey]
        r = requests.get(url)
        if r.status_code == 200:
            return self._parse_full_list_response(r)

    def _parse_full_list_response(self, response):
        soup = BeautifulSoup(response.text, features='html.parser')
        full_list = []
        skip_lines = 5
        for line in soup.get_text().split('\n')[skip_lines-1:]:
            if line == '':
                continue
            accession_number = line.split()[0]
            title = ' '.join(line.split()[1:])
            full_list.append((accession_number, title))
        return full_list

    def _parse_search_response(self, response):
        soup = BeautifulSoup(response.text, features='html.parser')
        divs = (x for x in soup.find_all('div'))
        results = []
        for div in divs:
            if div.a:
                name = div.a.get_text()
                next_div = next(divs)
                text = next_div.get_text()
                results.append((name, text.strip()))
        return results

    def get(self, accession_number, dbkey='aaindex'):
        params = ':'.join([dbkey, accession_number])
        url = ''.join([self.record_url, params])
        r = requests.get(url)
        if r.status_code == 200:
            new_record = Record(accession_number).from_response(r.text)
            return new_record


def string2NaN(x):
    try:
        x = float(x)
    except:
        x = np.nan
    return x

class Record:

    response_data = ''

    def __init__(self, record_id):
        self.record_id = record_id

    def from_response(self, response):
        soup = BeautifulSoup(response, features='html.parser')
        self.response_data = soup.find_all('pre').pop().get_text().strip()
        if len(self.response_data.split('\n')) <= 1:
            raise FileNotFoundError(f'{self.record_id}: No such data was found.')
        return self

    @property
    def accession_number(self):
        return self._rip_data('H')

    @property
    def data_description(self):
        return self._rip_data('D')

    @property
    def pmid(self):
        return self._rip_data('R')

    @property
    def author(self):
        return self._rip_data('A')

    @property
    def title(self):
        return self._rip_data('T')

    @property
    def journal_reference(self):
        return self._rip_data('J')

    @property
    def similar_entities(self):
        acn = (x for x in self._rip_data('C').split())
        data = [(x, float(next(acn))) for x in acn]
        return data

    @property
    def index_data(self):
        '''
        I: stands for a index
        M: stands for a matrix
        '''
        if self._rip_data('I'):
            idx_data = self._rip_data('I').split()
            data = {}
            for i in range(10):
                a1 = idx_data[i].split('/')[0]
                a2 = idx_data[i].split('/')[1]
                v1 = idx_data[i+10]
                v2 = idx_data[i+20]
                data[a1] = v1
                data[a2] = v2
            df = pd.Series(data).to_frame(name=self.record_id)
            
        if self._rip_data('M'):
            data = self._rip_data('M').split()
            rows = data[2].replace(',','')
            cols = data[5].replace(',','')
            rows = pd.Series(list(rows))
            cols = pd.Series(list(cols))
            values = data[6:]
            if len(values) == len(rows)*len(cols):
                df = pd.DataFrame(np.array(values).reshape(len(rows), len(cols)), index = rows, columns = cols)
                df = df.stack().reset_index()
                df.index = df.level_0 + ',' + df.level_1
                df = df[0].to_frame(name = self.record_id)
            else:
                idxs = []
                idxs2= []
                for i, row in enumerate(rows):
                    for col in cols[i:]:
                        idx = '%s,%s' % (row,col)
                        idx2 = '%s,%s' % (col, row)
                        idxs.append(idx)
                        idxs2.append(idx2)
                df1 = pd.DataFrame(values, index = idxs, columns = [self.record_id])
                df2 = pd.DataFrame(values, index = idxs2, columns = [self.record_id])
                df = df1.append(df2)
                df = df[~df.index.duplicated()]
                
        df[self.record_id] = df[self.record_id].apply(string2NaN)
        df.index.name = 'key'
        return df

    def _rip_data(self, flag):
        data = []
        line_generator = (x for x in self.response_data.split('\n'))
        for line in line_generator:
            if line.startswith(flag):
                data.extend(line.split()[1:])
                while True:
                    next_line = next(line_generator)
                    if next_line.startswith(' '):
                        data.extend(next_line.split())
                    else:
                        break
        return ' '.join(data)
