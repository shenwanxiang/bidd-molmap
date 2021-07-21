# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 18:07:26 2021

@Orignal author: Dr. Cao Dong Sheng (https://faculty.csu.edu.cn/caodongsheng/en/index.htm)

@updated by: Shen Wan Xiang 

"""



"""
#########################################################################################

Instead of using the conventional 20-D amino acid composition to represent the sample

of a protein, Prof. Kuo-Chen Chou proposed the pseudo amino acid (PseAA) composition 

in order for inluding the sequence-order information. Based on the concept of Chou's 
 
pseudo amino acid composition, the server PseAA was designed in a flexible way, allowing 
 
users to generate various kinds of pseudo amino acid composition for a given protein
 
sequence by selecting different parameters and their combinations. This module aims at 
 
computing two types of PseAA descriptors: Type I and Type II. 
 
You can freely use and distribute it. If you have any problem, you could contact 
 
with us timely.

References:

[1]: Kuo-Chen Chou. Prediction of Protein Cellular Attributes Using Pseudo-Amino Acid 

Composition. PROTEINS: Structure, Function, and Genetics, 2001, 43: 246-255.

[2]: http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/

[3]: http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/type2.htm

[4]: Kuo-Chen Chou. Using amphiphilic pseudo amino acid composition to predict enzyme 

subfamily classes. Bioinformatics, 2005,21,10-19.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2012.9.2

Email: oriental-cds@163.com


The hydrophobicity values are from JACS, 1962, 84: 4240-4246. (C. Tanford).

The hydrophilicity values are from PNAS, 1981, 78:3824-3828 (T.P.Hopp & K.R.Woods).

The side-chain mass for each of the 20 amino acids.

CRC Handbook of Chemistry and Physics, 66th ed., CRC Press, Boca Raton, Florida (1985).

R.M.C. Dawson, D.C. Elliott, W.H. Elliott, K.M. Jones, Data for Biochemical Research 3rd ed., 

Clarendon Press Oxford (1986).

#########################################################################################
"""

import math
#import scipy

AALetter=["A","R","N","D","C","E","Q","G","H","I",
          "L","K","M","F","P","S","T","W","Y","V"]

_Hydrophobicity={"A":0.62,"R":-2.53,"N":-0.78,"D":-0.90,"C":0.29,"Q":-0.85,
                 "E":-0.74,"G":0.48,"H":-0.40,"I":1.38,"L":1.06,"K":-1.50,
                 "M":0.64,"F":1.19,"P":0.12,"S":-0.18,"T":-0.05,"W":0.81,"Y":0.26,"V":1.08}

_hydrophilicity={"A":-0.5,"R":3.0,"N":0.2,"D":3.0,"C":-1.0,"Q":0.2,"E":3.0,
                 "G":0.0,"H":-0.5,"I":-1.8,"L":-1.8,"K":3.0,"M":-1.3,
                 "F":-2.5,"P":0.0,"S":0.3,"T":-0.4,"W":-3.4,"Y":-2.3,"V":-1.5}

_residuemass={"A":15.0,"R":101.0,"N":58.0,"D":59.0,"C":47.0,"Q":72.0,
              "E":73.0,"G":1.000,"H":82.0,"I":57.0,"L":57.0,"K":73.0,
              "M":75.0,"F":91.0,"P":42.0,"S":31.0,"T":45.0,"W":130.0,"Y":107.0,"V":43.0}

_pK1={"A":2.35,"C":1.71,"D":1.88,"E":2.19,"F":2.58,"G":2.34,"H":1.78,
      "I":2.32,"K":2.20,"L":2.36,"M":2.28,"N":2.18,"P":1.99,"Q":2.17,
      "R":2.18,"S":2.21,"T":2.15,"V":2.29,"W":2.38,"Y":2.20}

_pK2={"A":9.87,"C":10.78,"D":9.60,"E":9.67,"F":9.24,"G":9.60,"H":8.97,
      "I":9.76,"K":8.90,"L":9.60,"M":9.21,"N":9.09,"P":10.6,"Q":9.13,
      "R":9.09,"S":9.15,"T":9.12,"V":9.74,"W":9.39,"Y":9.11}

_pI={"A":6.11,"C":5.02,"D":2.98,"E":3.08,"F":5.91,"G":6.06,"H":7.64,
     "I":6.04,"K":9.47,"L":6.04,"M":5.74,"N":10.76,"P":6.30,
     "Q":5.65,"R":10.76,"S":5.68,"T":5.60,"V":6.02,"W":5.88,"Y":5.63}
#############################################################################################

def _mean(listvalue):
	"""
	########################################################################################
	The mean value of the list data.
	
	Usage:
	
	result=_mean(listvalue)
	########################################################################################
	"""
	return sum(listvalue)/len(listvalue)
##############################################################################################
def _std(listvalue,ddof=1):
	"""
	########################################################################################
	The standard deviation of the list data.
	
	Usage:
	
	result=_std(listvalue)
	########################################################################################
	"""
	mean=_mean(listvalue)
	temp=[math.pow(i-mean,2) for i in listvalue]
	res=math.sqrt(sum(temp)/(len(listvalue)-ddof))
	return res
##############################################################################################
def NormalizeEachAAP(AAP):
	"""
	########################################################################################
	All of the amino acid indices are centralized and 
	
	standardized before the calculation.
	
	Usage:
	
	result=NormalizeEachAAP(AAP)
	
	Input: AAP is a dict form containing the properties of 20 amino acids.
	
	Output: result is the a dict form containing the normalized properties 
	
	of 20 amino acids.
	########################################################################################
	"""
	if len(AAP.values())!=20:
		print('You can not input the correct number of properities of Amino acids!')
	else:
		Result={}
		for i,j in AAP.items():
			Result[i]=(j-_mean(AAP.values()))/_std(AAP.values(),ddof=0)

	return Result
#############################################################################################
#############################################################################################
##################################Type I descriptors#########################################
####################### Pseudo-Amino Acid Composition descriptors############################
#############################################################################################
#############################################################################################
def _GetCorrelationFunction(Ri='S',Rj='D',AAP=[_Hydrophobicity,_hydrophilicity,_residuemass]):
	"""
	########################################################################################
	Computing the correlation between two given amino acids using the above three
	
	properties.
	
	Usage:
	
	result=_GetCorrelationFunction(Ri,Rj)
	
	Input: Ri and Rj are the amino acids, respectively.
	
	Output: result is the correlation value between two amino acids.
	########################################################################################
	"""
	Hydrophobicity=NormalizeEachAAP(AAP[0])
	hydrophilicity=NormalizeEachAAP(AAP[1])
	residuemass=NormalizeEachAAP(AAP[2])
	theta1=math.pow(Hydrophobicity[Ri]-Hydrophobicity[Rj],2)
	theta2=math.pow(hydrophilicity[Ri]-hydrophilicity[Rj],2)
	theta3=math.pow(residuemass[Ri]-residuemass[Rj],2)
	theta=round((theta1+theta2+theta3)/3.0,3)
	return theta
#############################################################################################

def _GetSequenceOrderCorrelationFactor(ProteinSequence,k=1):
	"""
	########################################################################################
	Computing the Sequence order correlation factor with gap equal to k based on 
	
	[_Hydrophobicity,_hydrophilicity,_residuemass].
	
	Usage:
	
	result=_GetSequenceOrderCorrelationFactor(protein,k)
	
	Input: protein is a pure protein sequence.
	
	k is the gap.
	
	Output: result is the correlation factor value with the gap equal to k.
	########################################################################################
	"""
	LengthSequence=len(ProteinSequence)
	res=[]
	for i in range(LengthSequence-k):
		AA1=ProteinSequence[i]
		AA2=ProteinSequence[i+k]
		res.append(_GetCorrelationFunction(AA1,AA2))
	result=round(sum(res)/(LengthSequence-k),3)
	return result
#############################################################################################

def GetAAComposition(ProteinSequence):

	"""
	########################################################################################
	Calculate the composition of Amino acids 
	
	for a given protein sequence.
	
	Usage:
	
	result=CalculateAAComposition(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing the composition of 
	
	20 amino acids.
	########################################################################################
	"""
	LengthSequence=len(ProteinSequence)
	Result={}
	for i in AALetter:
		Result[i]=round(float(ProteinSequence.count(i))/LengthSequence*100,3)
	return Result

#############################################################################################
def _GetPseudoAAC1(ProteinSequence,lamda=10,weight=0.05):
	"""
	#######################################################################################
	Computing the first 20 of type I pseudo-amino acid compostion descriptors based on
	
	[_Hydrophobicity,_hydrophilicity,_residuemass].
	########################################################################################
	"""
	rightpart=0.0
	for i in range(lamda):
		rightpart=rightpart+_GetSequenceOrderCorrelationFactor(ProteinSequence,k=i+1)
	AAC=GetAAComposition(ProteinSequence)
	
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['PAAC'+str(index+1)]=round(AAC[i]/temp,3)
	
	return result

#############################################################################################
def _GetPseudoAAC2(ProteinSequence,lamda=10,weight=0.05):
	"""
	########################################################################################
	Computing the last lamda of type I pseudo-amino acid compostion descriptors based on
	
	[_Hydrophobicity,_hydrophilicity,_residuemass].
	########################################################################################
	"""
	rightpart=[]
	for i in range(lamda):
		rightpart.append(_GetSequenceOrderCorrelationFactor(ProteinSequence,k=i+1))
	
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+lamda):
		result['PAAC'+str(index+1)]=round(weight*rightpart[index-20]/temp*100,3)
	
	return result
#############################################################################################

def _GetPseudoAAC(ProteinSequence,lamda=10,weight=0.05):
	"""
	#######################################################################################
	Computing all of type I pseudo-amino acid compostion descriptors based on three given
	
	properties. Note that the number of PAAC strongly depends on the lamda value. if lamda 
	
	= 20, we can obtain 20+20=40 PAAC descriptors. The size of these values depends on the 
	
	choice of lamda and weight simultaneously. 
	
	AAP=[_Hydrophobicity,_hydrophilicity,_residuemass]
	
	Usage:
	
	result=_GetAPseudoAAC(protein,lamda,weight)
	
	Input: protein is a pure protein sequence.
	
	lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15.
	
	Note that (1)lamda should NOT be larger than the length of input protein sequence;
	
	(2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the 
	
	output of PseAA server is the 20-D amino acid composition.
	
	weight factor is designed for the users to put weight on the additional PseAA components 
	
	with respect to the conventional AA components. The user can select any value within the 
	
	region from 0.05 to 0.7 for the weight factor.
	
	Output: result is a dict form containing calculated 20+lamda PAAC descriptors.
	########################################################################################
	"""
	res={}
	res.update(_GetPseudoAAC1(ProteinSequence,lamda=lamda,weight=weight))
	res.update(_GetPseudoAAC2(ProteinSequence,lamda=lamda,weight=weight))
	return res
#############################################################################################
##################################Type II descriptors########################################
###############Amphiphilic Pseudo-Amino Acid Composition descriptors#########################
#############################################################################################
#############################################################################################
def _GetCorrelationFunctionForAPAAC(Ri='S',Rj='D', AAP=[_Hydrophobicity,_hydrophilicity]):
	"""
	########################################################################################
	Computing the correlation between two given amino acids using the above two
	
	properties for APAAC (type II PseAAC).
	
	Usage:
	
	result=_GetCorrelationFunctionForAPAAC(Ri,Rj)
	
	Input: Ri and Rj are the amino acids, respectively.
	
	Output: result is the correlation value between two amino acids.
	########################################################################################
	"""
	Hydrophobicity=NormalizeEachAAP(AAP[0])
	hydrophilicity=NormalizeEachAAP(AAP[1])
	theta1=round(Hydrophobicity[Ri]*Hydrophobicity[Rj],3)
	theta2=round(hydrophilicity[Ri]*hydrophilicity[Rj],3)

	return theta1,theta2


#############################################################################################
def GetSequenceOrderCorrelationFactorForAPAAC(ProteinSequence, 
                                              k=1, 
                                              AAP = [_Hydrophobicity,_hydrophilicity]):
    
	"""
	########################################################################################
	Computing the Sequence order correlation factor with gap equal to k based on 
	
	[_Hydrophobicity,_hydrophilicity] for APAAC (type II PseAAC) .
	
	Usage:
	
	result=GetSequenceOrderCorrelationFactorForAPAAC(protein,k)
	
	Input: protein is a pure protein sequence.
	
	k is the gap.
	
	Output: result is the correlation factor value with the gap equal to k.
	########################################################################################
	"""
	LengthSequence=len(ProteinSequence)
	resHydrophobicity=[]
	reshydrophilicity=[]
	for i in range(LengthSequence-k):
		AA1=ProteinSequence[i]
		AA2=ProteinSequence[i+k]
		temp=_GetCorrelationFunctionForAPAAC(AA1,AA2, AAP=AAP)
		resHydrophobicity.append(temp[0])
		reshydrophilicity.append(temp[1])
	result=[]
	result.append(round(sum(resHydrophobicity)/(LengthSequence-k),3))
	result.append(round(sum(reshydrophilicity)/(LengthSequence-k),3))
	return result
#############################################################################################
def GetAPseudoAAC1(ProteinSequence,lamda=30,weight=0.5, AAP = [_Hydrophobicity,_hydrophilicity]):
	"""
	########################################################################################
	Computing the first 20 of type II pseudo-amino acid compostion descriptors based on
	
	[_Hydrophobicity,_hydrophilicity].
	########################################################################################
	"""
	rightpart=0.0
	for i in range(lamda):
		rightpart=rightpart+sum(GetSequenceOrderCorrelationFactorForAPAAC(ProteinSequence,k=i+1, AAP=AAP))
	AAC=GetAAComposition(ProteinSequence)
	
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['APAAC'+str(index+1)]=round(AAC[i]/temp,3)
	
	return result

#############################################################################################
def GetAPseudoAAC2(ProteinSequence,lamda=30,weight=0.5, AAP = [_Hydrophobicity,_hydrophilicity]):
	"""
	#######################################################################################
	Computing the last lamda of type II pseudo-amino acid compostion descriptors based on
	
	[_Hydrophobicity,_hydrophilicity].
	#######################################################################################
	"""
	rightpart=[]
	for i in range(lamda):
		temp=GetSequenceOrderCorrelationFactorForAPAAC(ProteinSequence,k=i+1, AAP=AAP)
		rightpart.append(temp[0])
		rightpart.append(temp[1])
		
	
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+2*lamda):
		result['PAAC'+str(index+1)]=round(weight*rightpart[index-20]/temp*100,3)
	
	return result
	
#############################################################################################
def GetAPseudoAAC(ProteinSequence,lamda=30,weight=0.5, AAP = [_Hydrophobicity,_hydrophilicity]):
	"""
	#######################################################################################
	Computing all of type II pseudo-amino acid compostion descriptors based on the given 
	
	properties. Note that the number of PAAC strongly depends on the lamda value. if lamda 
	
	= 20, we can obtain 20+20=40 PAAC descriptors. The size of these values depends on the 
	
	choice of lamda and weight simultaneously.
	
	Usage:
	
	result=GetAPseudoAAC(protein,lamda,weight)
	
	Input: protein is a pure protein sequence.
	
	lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15.
	
	Note that (1)lamda should NOT be larger than the length of input protein sequence;
	
	(2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the 
	
	output of PseAA server is the 20-D amino acid composition.
	
	weight factor is designed for the users to put weight on the additional PseAA components 
	
	with respect to the conventional AA components. The user can select any value within the 
	
	region from 0.05 to 0.7 for the weight factor.
	
	Output: result is a dict form containing calculated 20+lamda PAAC descriptors.
	#######################################################################################
	"""
	res={}
	res.update(GetAPseudoAAC1(ProteinSequence,lamda=lamda,weight=weight, AAP = AAP))
	res.update(GetAPseudoAAC2(ProteinSequence,lamda=lamda,weight=weight, AAP = AAP))
	return res
#############################################################################################
#############################################################################################
##################################Type I descriptors#########################################
####################### Pseudo-Amino Acid Composition descriptors############################
#############################based on different properties###################################
#############################################################################################
#############################################################################################
def GetCorrelationFunction(Ri='S',Rj='D',AAP=[]):
	"""
	########################################################################################
	Computing the correlation between two given amino acids using the given
	
	properties.
	
	Usage:
	
	result=GetCorrelationFunction(Ri,Rj,AAP)
	
	Input: Ri and Rj are the amino acids, respectively.
	
	AAP is a list form containing the properties, each of which is a dict form.
	
	Output: result is the correlation value between two amino acids.
	########################################################################################
	"""
	NumAAP=len(AAP)
	theta=0.0
	for i in range(NumAAP):
		temp=NormalizeEachAAP(AAP[i])
		theta=theta+math.pow(temp[Ri]-temp[Rj],2)
	result=round(theta/NumAAP,3)
	return result
#############################################################################################
def GetSequenceOrderCorrelationFactor(ProteinSequence,k=1,AAP=[]):
	"""
	########################################################################################
	Computing the Sequence order correlation factor with gap equal to k based on 
	
	the given properities.
	
	Usage:
	
	result=GetSequenceOrderCorrelationFactor(protein,k,AAP)
	
	Input: protein is a pure protein sequence.
	
	k is the gap.
	
	AAP is a list form containing the properties, each of which is a dict form.
	
	Output: result is the correlation factor value with the gap equal to k.
	########################################################################################
	"""
	LengthSequence=len(ProteinSequence)
	res=[]
	for i in range(LengthSequence-k):
		AA1=ProteinSequence[i]
		AA2=ProteinSequence[i+k]
		res.append(GetCorrelationFunction(AA1,AA2,AAP))
	result=round(sum(res)/(LengthSequence-k),3)
	return result
#############################################################################################
def GetPseudoAAC1(ProteinSequence,lamda=30,weight=0.05,AAP=[_Hydrophobicity,_hydrophilicity]):
	"""
	#######################################################################################
	Computing the first 20 of type I pseudo-amino acid compostion descriptors based on the given 
	
	properties.
	########################################################################################
	"""
	rightpart=0.0
	for i in range(lamda):
		rightpart=rightpart+GetSequenceOrderCorrelationFactor(ProteinSequence,i+1,AAP)
	AAC=GetAAComposition(ProteinSequence)
	
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['PAAC'+str(index+1)]=round(AAC[i]/temp,3)
	
	return result

#############################################################################################
def GetPseudoAAC2(ProteinSequence,lamda=30,weight=0.05,AAP=[_Hydrophobicity,_hydrophilicity]):
	"""
	#######################################################################################
	Computing the last lamda of type I pseudo-amino acid compostion descriptors based on the given 
	
	properties.
	########################################################################################
	"""
	rightpart=[]
	for i in range(lamda):
		rightpart.append(GetSequenceOrderCorrelationFactor(ProteinSequence,i+1,AAP))
	
	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+lamda):
		result['PAAC'+str(index+1)]=round(weight*rightpart[index-20]/temp*100,3)
	
	return result
#############################################################################################

def GetPseudoAAC(ProteinSequence,lamda=30,weight=0.05,AAP=[_Hydrophobicity,_hydrophilicity]):
	"""
	#######################################################################################
	Computing all of type I pseudo-amino acid compostion descriptors based on the given 
	
	properties. Note that the number of PAAC strongly depends on the lamda value. if lamda 
	
	= 20, we can obtain 20+20=40 PAAC descriptors. The size of these values depends on the 
	
	choice of lamda and weight simultaneously. You must specify some properties into AAP.
	
	Usage:
	
	result=GetPseudoAAC(protein,lamda,weight)
	
	Input: protein is a pure protein sequence.
	
	lamda factor reflects the rank of correlation and is a non-Negative integer, such as 15.
	
	Note that (1)lamda should NOT be larger than the length of input protein sequence;
	
	(2) lamda must be non-Negative integer, such as 0, 1, 2, ...; (3) when lamda =0, the 
	
	output of PseAA server is the 20-D amino acid composition.
	
	weight factor is designed for the users to put weight on the additional PseAA components 
	
	with respect to the conventional AA components. The user can select any value within the 
	
	region from 0.05 to 0.7 for the weight factor.
	
	AAP is a list form containing the properties, each of which is a dict form.
	
	Output: result is a dict form containing calculated 20+lamda PAAC descriptors.
	########################################################################################
	"""
	res={}
	res.update(GetPseudoAAC1(ProteinSequence,lamda,weight,AAP))
	res.update(GetPseudoAAC2(ProteinSequence,lamda,weight,AAP))
	return res
#############################################################################################

if __name__=="__main__":

	protein="MTDRARLRLHDTAAGVVRDFVPLRPGHVSIYLCGATVQGLPHIGHVRSGVAFDILRRWLL"
    
    #lamda should be smaller that length of the seq
    
	#PAAC=GetPseudoAAC(protein,lamda=5,AAP=[_Hydrophobicity,_hydrophilicity])
	PAAC=GetPseudoAAC(protein,lamda=30, weight=0.05, ) #type-1
	APAAC = GetAPseudoAAC(protein,lamda=30, weight=0.5) #type-2
