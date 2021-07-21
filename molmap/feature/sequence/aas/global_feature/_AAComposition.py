# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 18:07:26 2021

@Orignal author: Dr. Cao Dong Sheng (https://faculty.csu.edu.cn/caodongsheng/en/index.htm)

@updated by: Shen Wan Xiang (py2 to py3, add Calculate2AACon3AA fuction) 


"""

"""
###############################################################################

The module is used for computing the composition of amino acids, dipetide and 

3-mers (tri-peptide) for a given protein sequence. You can get 8420 descriptors 

for a given protein sequence. You can freely use and distribute it. If you hava 

any problem, you could contact with us timely!

References:

[1]: Reczko, M. and Bohr, H. (1994) The DEF data base of sequence based protein

fold class predictions. Nucleic Acids Res, 22, 3616-3619.

[2]: Hua, S. and Sun, Z. (2001) Support vector machine approach for protein

subcellular localization prediction. Bioinformatics, 17, 721-728.


[3]:Grassmann, J., Reczko, M., Suhai, S. and Edler, L. (1999) Protein fold class

prediction: new methods of statistical classification. Proc Int Conf Intell Syst Mol

Biol, 106-112.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2012.3.27

Email: oriental-cds@163.com

###############################################################################
"""

import re
import pandas as pd



AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
#############################################################################################
def CalculateAAComposition(ProteinSequence):

	"""
	########################################################################
	Calculate the composition of Amino acids 
	
	for a given protein sequence.
	
	Usage:
	
	result=CalculateAAComposition(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing the composition of 
	
	20 amino acids.
	########################################################################
	"""
	LengthSequence=len(ProteinSequence)
	Result={}
	for i in AALetter:
		Result[i]=round(float(ProteinSequence.count(i))/LengthSequence*100,3)
	return Result

#############################################################################################
def CalculateDipeptideComposition(ProteinSequence):
	"""
	########################################################################
	Calculate the composition of dipeptidefor a given protein sequence.
	
	Usage: 
	
	result=CalculateDipeptideComposition(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing the composition of 
	
	400 dipeptides.
	########################################################################
	"""

	LengthSequence=len(ProteinSequence)
	Result={}
	for i in AALetter:
		for j in AALetter:
			Dipeptide=i+j
			Result[Dipeptide]=round(float(ProteinSequence.count(Dipeptide))/(LengthSequence-1)*100,2)
	return Result



#############################################################################################

def Getkmers():
	"""
	########################################################################
	Get the amino acid list of 3-mers. 
	
	Usage: 
	
	result=Getkmers()
	
	Output: result is a list form containing 8000 tri-peptides.
	
	########################################################################
	"""
	kmers=list()
	for i in AALetter:
		for j in AALetter:
			for k in AALetter:
				kmers.append(i+j+k)
	return kmers

#############################################################################################
def GetSpectrumDict(proteinsequence):
	"""
	########################################################################
	Calcualte the spectrum descriptors of 3-mers for a given protein.
	
	Usage: 
	
	result=GetSpectrumDict(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing the composition values of 8000
	
	3-mers.
	########################################################################
	"""
	result={}
	kmers=Getkmers()
	for i in kmers:
		result[i]=len(re.findall(i,proteinsequence))
	return result



#############################################################################################
def _replace(x):
	#x = 'ACA'
	x = list(x)
	x[1] = '*'
	x = ''.join(x)
	return x

def Calculate2AACon3AA(proteinsequence):
	"""
	########################################################################
	Calcualte the composition descriptors of 3-mers for a given protein, note that the middle aa in the 3-mer is ignored
	
	Usage: 
	
	result=Calculate2AACon3AA(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing the composition values of 400 3-mers (ignore middle).
	########################################################################
	"""
	res = GetSpectrumDict(proteinsequence)
	s1 = pd.Series(res)
	s1.index = s1.index.map(_replace)
	res = (s1.reset_index().groupby('index')[0].sum()/len(proteinsequence)*100).round(3).to_dict()
	return res


#############################################################################################
def CalculateAADipeptideComposition(ProteinSequence):

	"""
	########################################################################
	Calculate the composition of AADs, dipeptide and 3-mers for a 
	
	given protein sequence.
	
	Usage:
	
	result=CalculateAADipeptideComposition(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing all composition values of 
	
	AADs, dipeptide and 3-mers (8420).
	########################################################################
	"""

	result={}
	result.update(CalculateAAComposition(ProteinSequence))
	result.update(CalculateDipeptideComposition(ProteinSequence))
	result.update(GetSpectrumDict(ProteinSequence))

	return result
#############################################################################################
if __name__=="__main__":

	protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
	AAC=CalculateAAComposition(protein)
	DAAC=CalculateDipeptideComposition(protein)
	spectrum=GetSpectrumDict(protein)
	TAAC=Calculate2AACon3AA(protein)
