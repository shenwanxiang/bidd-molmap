#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:52:49 2018

@author: shenwanxiang

Multi process Run
"""

import time
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, wait, as_completed
from multiprocessing import Pool,cpu_count,current_process 
import subprocess


from molmap.utils.logtools import print_info, print_error, pbar,print_warn


def RunCmd(cmd):
    '''
    input:
        cmd: str
    output:
        status: int, 0 for success
        stdout: str
        stderr: str
        
    '''
    print_info('run command : %s' % cmd)
    
    def swap_log(swap, error = True):
        sinfo = []
        for l in swap.split('\n'):
            if l == '':
                continue
            sinfo.append(l)
        for o in sinfo:
            if error:
                print_error(o) 
            else:
                print_info(o) 
        return            
    output = subprocess.run(cmd, 
                            shell=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            universal_newlines=True)    
    status = output.returncode
    stdout = output.stdout
    stderr = output.stderr
    
    if status != 0:
        if output.stdout:
             swap_log(output.stdout, error=True)
        if output.stderr:
             swap_log(output.stderr, error=True)
    else:
        if output.stdout:
            swap_log(output.stdout, error=False)
    #return status

    return status, stdout, stderr



def ImapUnorder(processor, iterator, max_workers=10, fail_in_file = './filed.lst'):
    '''
    processor: fuction
    iterator: list or iterator,each element should be a tuple or dict, so that data can be used as ordered 
    '''
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        with open(fail_in_file, 'w+') as f:
            futures = {executor.submit(processor, IdPlusSmile):IdPlusSmile for IdPlusSmile in iterator}
            success, _ = wait(futures)
            with pbar(total = len(futures)) as pb:
                for i in success:
                    IdPlusSmile = futures[i]
                    print_info('deal '+ str(IdPlusSmile))
                    try:
                        data_dict = i.result()
                        yield data_dict
                    except Exception as exc:
                        print_warn('because of the process is dead, input: %s is fialed when deal with %s: %s, so we will deal it automatically' % (IdPlusSmile, processor, exc))
                        
                        try: 
                            yield processor(IdPlusSmile)
                        except:
                            f.write(str(IdPlusSmile)+'\n')
                            print_error(' input: %s is fialed when deal with %s: %s' % (IdPlusSmile, processor, exc))
                    pb.update(1)

                    


                    
def MultiProcessUnorderedBarRun(func, deal_list, n_cpus=None):
    if n_cpus ==None:
        N_CPUS = cpu_count()
    else:
        N_CPUS = int(n_cpus)
    print_info('the number of process is %s' % N_CPUS)
    
    p = Pool(N_CPUS)
    res_list = []
    with pbar(total = len(deal_list)) as pb:
        for res in p.imap_unordered(func, deal_list):
            pb.update(1)
            res_list.append(res)
    p.close()
    p.join()
    return res_list



def MultiProcessRun(func, deal_list, n_cpus=None):
    
    '''
    input:
        func: function to do with each element in the deal_list
        deal_list: list to be done
        n_cpus: use the number of cpus
    output:
        list of the return result for each func
    '''
    
    #round_c = [deal_list[i:i+batch_size] for i  in range(0, len(deal_list), batch_size)]
    #mata thinking: https://my.oschina.net/leejun2005/blog/203148
    if n_cpus ==None:
        N_CPUS = cpu_count()
    else:
        N_CPUS = int(n_cpus)

    print_info('the number of process is %s' % N_CPUS)

    pool = Pool(N_CPUS)
    a = pool.map(func, deal_list)
    pool.close()
    pool.join()
    return a






########### ordered map reduce  ##############
def _decorate_func(func, i, j):
    return [i, func(j)]

def _executor(func, series, n_cpus = 4):
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        futures = [executor.submit(_decorate_func, func, i, j) for i,j in series.iteritems()]
    return futures


def MultiExecutorRun(func, deal_list, n_cpus = 4, tqdm_args = {'unit':'one'}):
    
    '''
    input:
        func: function to do with each element in the deal_list
        deal_list: list to be done
        n_cpus: use the number of cpus
        tqdm_args: args for tqdm
    output:
        list of the return value for each func
    '''
    lst  =list(deal_list)
    series = pd.Series(lst)
    
    futures = _executor(func, series, n_cpus = n_cpus)
    args = {
        'total': len(deal_list),
        'unit': 'one',
        'ascii': True,
        'unit_scale': True,
        'leave': True
    }
    args.update(tqdm_args)
    
    print_info(args)
    
    results = []
    indexs = []
    for f in tqdm(as_completed(futures), **args):
        #print(f)
        idx, result = f.result()
        indexs.append(idx)
        results.append(result)
    
    res = pd.Series(results,index=indexs)
    #sort unordered result
    ordered_lst = res.sort_index().tolist()
    return ordered_lst
    