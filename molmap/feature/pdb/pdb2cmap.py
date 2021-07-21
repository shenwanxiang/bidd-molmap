# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:37:10 2021

@author: Wanxiang Shen
"""

import sys,os
from math import sqrt
import numpy as np

valid_amino_acids = {
    'LLP': 'K', 'TPO': 'T', 'CSS': 'C', 'OCS': 'C', 'CSO': 'C', 'PCA': 'E', 'KCX': 'K', \
    'CME': 'C', 'MLY': 'K', 'SEP': 'S', 'CSX': 'C', 'CSD': 'C', 'MSE': 'M', \
    'ALA': 'A', 'ASN': 'N', 'CYS': 'C', 'GLN': 'Q', 'HIS': 'H', 'LEU': 'L', \
    'MET': 'M', 'MHO': 'M', 'PRO': 'P', 'THR': 'T', 'TYR': 'Y', 'ARG': 'R', 'ASP': 'D', \
    'GLU': 'E', 'GLY': 'G', 'ILE': 'I', 'LYS': 'K', 'PHE': 'F', 'SER': 'S', \
    'TRP': 'W', 'VAL': 'V', 'SEC': 'U'
    }


def check_pdb_valid_row(valid_amino_acids, l):
    if (get_pdb_rname(l) in valid_amino_acids.keys()) and (l.startswith('ATOM') or l.startswith('HETA')):
        return True
    return False

def get_pdb_atom_name(l):
    return l[12: 16].strip()

def get_pdb_rnum(l):
    return int(l[22: 27].strip())


def get_pdb_rname(l):
    return l[17: 20].strip()


def get_pdb_xyz_cb(lines):
    xyz = {}
    for l in lines:
        if not (l.startswith('ATOM') or l.startswith('HE')):
            continue
        if get_pdb_atom_name(l) == 'CB':
            xyz[get_pdb_rnum(l)] = (float(l[30:38].strip()), float(l[38:46].strip()), float(l[46:54].strip()))
    for l in lines:
        if not (l.startswith('ATOM') or l.startswith('HE')):
            continue
        if (get_pdb_rnum(l) not in xyz) and get_pdb_atom_name(l) == 'CA':
            xyz[get_pdb_rnum(l)] = (float(l[30:38].strip()), float(l[38:46].strip()), float(l[46:54].strip()))
    return xyz

def get_pdb_xyz_ca(lines):
    xyz = {}
    for l in lines:
        if not (l.startswith('ATOM') or l.startswith('HE')):
            continue
        if get_pdb_atom_name(l) == 'CA':
            xyz[get_pdb_rnum(l)] = (float(l[30:38].strip()), float(l[38:46].strip()), float(l[46:54].strip()))
    return xyz

def parse_pdb_row(row, param):
    result = ''
    if param == 'rnum':
        result = row[22:27]
        if any(c.isalpha() for c in result):
            valid_list = {
                '   0A': '-999',
                '   0B': '-998',
                '   0C': '-997',
                '   0D': '-996',
                '   0E': '-995',
                '   0F': '-994',
                '   0G': '-993',
                '   0H': '-992',
                '   0I': '-991',
                '   0J': '-990',
            }
            if result in valid_list.keys():
                result = valid_list[result]
            else:
                #print('Alternative ' + row + ' - skipping..\n')
                return 'NA'
    if param == 'aname':
        result = row[12:16]
    if param == 'altloc':
        result = row[16:17]
    if param == 'rname':
        result = row[17:20]
    if param == 'chain':
        result = row[21:22]
    if param == 'x':
        result = row[30:38]
    if param == 'y':
        result = row[38:46]
    if param == 'z':
        result = row[46:54]
    if len(result) < 1:
        print('Error! Undefined param/result!')
        sys.exit(1)
    return result.strip()

def reindex_chain_from_rnum1(valid_amino_acids, inpdb, maxresnum = 512):

    atom_counter = 0;
    prev_res_num_in_inPDB = -10000;


    f = open(inpdb, mode = 'r')
    lines = f.read()
    f.close()
    lines = lines.splitlines()
    print('  input rows:', len(lines))

    # Skip the last lines of HETATM
    skip_from_this_line_on = '--';
    for line in reversed(lines):
        if line.startswith('HETATM') and 'MSE' not in line:
            skip_from_this_line_on = line
        else:
            break

    print('  skipping lines after: ', skip_from_this_line_on)
    new_rnum = 0;
    # skip all the residues that do not have CA
    residues_to_skip = {}
    for line in lines:
        if len(line) < 10:
            continue
        if not (line[16:17] == ' ' or line[16:17] == 'A'):
            continue
        this_rnum = parse_pdb_row(line, 'rnum')
        residues_to_skip[this_rnum] = 1
    for line in lines:
        if len(line) < 10:
            continue
        if not (line[16:17] == ' ' or line[16:17] == 'A'):
            continue
        this_rnum = parse_pdb_row(line, 'rnum')
        this_aname = parse_pdb_row(line, 'aname')
        if this_aname == 'CA' and this_rnum in residues_to_skip:
            del residues_to_skip[this_rnum]

    lines_to_write = []
    for line in lines:
        if len(line) < 10:
            continue
        if not (line[16:17] == ' ' or line[16:17] == 'A'):
            continue
        if line == skip_from_this_line_on:
            break
        this_rnum = parse_pdb_row(line, 'rnum')
        if this_rnum in residues_to_skip:
            continue
        this_rname = parse_pdb_row(line, 'rname')
        if this_rname not in valid_amino_acids.keys():
            continue
        if this_rnum == 'NA':
            continue
        if prev_res_num_in_inPDB != this_rnum:
            prev_res_num_in_inPDB = this_rnum
            new_rnum += 1
        if new_rnum == maxresnum + 1:
            break
        atom_counter += 1
        str_atom_counter = '%5s' % (atom_counter)
        str_new_rnum = '%4s' % (new_rnum)
        lines_to_write.append(line[:6] + str_atom_counter + line[11:16] + ' ' + line[17:20] + '  ' + str_new_rnum + ' ' + line[27:])

    print('  output rows:', len(lines_to_write))

    if (len(lines_to_write) < 10):
        print('WARNING! Too few lines to write.. \n')

    f = open(inpdb + '.tmp', mode = 'w')
    for line in lines_to_write:
        f.write(line + '\n')
    f.close()

    os.system('mv' + ' ' + inpdb + '.tmp' + ' ' + inpdb)
    return new_rnum



def get_dist_maps(file_pdb, flag_gaps=False, flag_any2any = False, valid_amino_acids = valid_amino_acids,):
    f = open(file_pdb, mode = 'r')
    lines = f.read()
    f.close()
    lines = lines.splitlines()
    rnum_rnames = {}
    for l in lines:
        atom = get_pdb_atom_name(l)
        if atom != 'CA':
            continue
        if not get_pdb_rname(l) in valid_amino_acids.keys():
            print ('' + get_pdb_rname(l) + ' is unknown amino acid in ' + l)
            sys.exit(1)
        rnum_rnames[int(get_pdb_rnum(l))] = valid_amino_acids[get_pdb_rname(l)]
    seq = ''
    for i in range(max(rnum_rnames.keys())):
        if i+1 not in rnum_rnames:
            print (rnum_rnames)
            print ('Warning! ' + file_pdb + ' ! residue not defined for rnum = ' + str(i+1))
            seq = seq + '-'
        else:
            seq = seq + rnum_rnames[i+1]
    L = len(seq)
    xyz_cb = get_pdb_xyz_cb(lines)

    if not flag_gaps:
        if len(xyz_cb) != L:
            print(rnum_rnames)
            for i in range(L):
                if i+1 not in xyz_cb:
                    print('XYZ not defined for ' + str(i+1))
            print ('Error! ' + file_pdb + ' Something went wrong - len of cbxyz != seqlen!! ' + str(len(xyz_cb)) + ' ' +  str(L))
            sys.exit(1)

    cb_map = np.full((L, L), np.nan)
    for r1 in sorted(xyz_cb):
        (a, b, c) = xyz_cb[r1]
        for r2 in sorted(xyz_cb):
            (p, q, r) = xyz_cb[r2]
            cb_map[r1 - 1, r2 - 1] = sqrt((a-p)**2+(b-q)**2+(c-r)**2)
    xyz_ca = get_pdb_xyz_ca(lines)

    if not flag_gaps:
        if len(xyz_ca) != L:
            print ('Something went wrong - len of cbxyz != seqlen!! ' + str(len(xyz_ca)) + ' ' +  str(L))
            sys.exit(1)

    ca_map = np.full((L, L), np.nan)
    for r1 in sorted(xyz_ca):
        (a, b, c) = xyz_ca[r1]
        for r2 in sorted(xyz_ca):
            (p, q, r) = xyz_ca[r2]
            ca_map[r1 - 1, r2 - 1] = sqrt((a-p)**2+(b-q)**2+(c-r)**2)

    if flag_any2any:
        any_map = np.full((L, L), np.nan)
        for l1 in lines:
            if not check_pdb_valid_row(check_pdb_valid_row, l1):
                continue
            r1 = get_pdb_rnum(l1)
            (a, b, c) = (float(l1[30:38].strip()), float(l1[38:46].strip()), float(l1[46:54].strip()))
            for l2 in lines:
                if not check_pdb_valid_row(check_pdb_valid_row, l2):
                    continue
                r2 = get_pdb_rnum(l2)
                (p, q, r) = (float(l2[30:38].strip()), float(l2[38:46].strip()), float(l2[46:54].strip()))
                d = sqrt((a-p)**2+(b-q)**2+(c-r)**2)
                if any_map[r1 - 1, r2 - 1] > d:
                    any_map[r1 - 1, r2 - 1] = d
        return L, seq, cb_map, ca_map, any_map
    return L, seq, cb_map, ca_map



if __name__ == '__main__':
    L, seq, cb_map, ca_map = get_dist_maps('./1a2zA.pdb')
