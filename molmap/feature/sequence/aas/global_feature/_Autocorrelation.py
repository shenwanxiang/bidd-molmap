# -*- coding: utf-8 -*-
"""
##########################################################################################

This module is used for computing the Autocorrelation descriptors based different

 properties of AADs.You can also input your properties of AADs, then it can help you

to compute Autocorrelation descriptors based on the property of AADs. Currently, You 

can get 720 descriptors for a given protein sequence based on our provided physicochemical

properties of AADs. You can freely use and distribute it. If you hava  any problem, 

you could contact with us timely!

References:

[1]: http://www.genome.ad.jp/dbget/aaindex.html

[2]:Feng, Z.P. and Zhang, C.T. (2000) Prediction of membrane protein types based on

the hydrophobic index of amino acids. J Protein Chem, 19, 269-275.

[3]:Horne, D.S. (1988) Prediction of protein helix content from an autocorrelation

analysis of sequence hydrophobicities. Biopolymers, 27, 451-477.

[4]:Sokal, R.R. and Thomson, B.A. (2006) Population structure inferred by local

spatial autocorrelation: an Usage from an Amerindian tribal population. Am J

Phys Anthropol, 129, 121-131.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2010.11.22

Email: oriental-cds@163.com

##########################################################################################
"""

import math


AALetter=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

_Hydrophobicity={"A":0.02,"R":-0.42,"N":-0.77,"D":-1.04,"C":0.77,"Q":-1.10,"E":-1.14,"G":-0.80,"H":0.26,"I":1.81,"L":1.14,"K":-0.41,"M":1.00,"F":1.35,"P":-0.09,"S":-0.97,"T":-0.77,"W":1.71,"Y":1.11,"V":1.13}

_AvFlexibility={"A":0.357,"R":0.529,"N":0.463,"D":0.511,"C":0.346,"Q":0.493,"E":0.497,"G":0.544,"H":0.323,"I":0.462,"L":0.365,"K":0.466,"M":0.295,"F":0.314,"P":0.509,"S":0.507,"T":0.444,"W":0.305,"Y":0.420,"V":0.386}

_Polarizability={"A":0.046,"R":0.291,"N":0.134,"D":0.105,"C":0.128,"Q":0.180,"E":0.151,"G":0.000,"H":0.230,"I":0.186,"L":0.186,"K":0.219,"M":0.221,"F":0.290,"P":0.131,"S":0.062,"T":0.108,"W":0.409,"Y":0.298,"V":0.140}

_FreeEnergy={"A":-0.368,"R":-1.03,"N":0.0,"D":2.06,"C":4.53,"Q":0.731,"E":1.77,"G":-0.525,"H":0.0,"I":0.791,"L":1.07,"K":0.0,"M":0.656,"F":1.06,"P":-2.24,"S":-0.524,"T":0.0,"W":1.60,"Y":4.91,"V":0.401}

_ResidueASA={"A":115.0,"R":225.0,"N":160.0,"D":150.0,"C":135.0,"Q":180.0,"E":190.0,"G":75.0,"H":195.0,"I":175.0,"L":170.0,"K":200.0,"M":185.0,"F":210.0,"P":145.0,"S":115.0,"T":140.0,"W":255.0,"Y":230.0,"V":155.0}

_ResidueVol={"A":52.6,"R":109.1,"N":75.7,"D":68.4,"C":68.3,"Q":89.7,"E":84.7,"G":36.3,"H":91.9,"I":102.0,"L":102.0,"K":105.1,"M":97.7,"F":113.9,"P":73.6,"S":54.9,"T":71.2,"W":135.4,"Y":116.2,"V":85.1}

_Steric={"A":0.52,"R":0.68,"N":0.76,"D":0.76,"C":0.62,"Q":0.68,"E":0.68,"G":0.00,"H":0.70,"I":1.02,"L":0.98,"K":0.68,"M":0.78,"F":0.70,"P":0.36,"S":0.53,"T":0.50,"W":0.70,"Y":0.70,"V":0.76}

_Mutability={"A":100.0,"R":65.0,"N":134.0,"D":106.0,"C":20.0,"Q":93.0,"E":102.0,"G":49.0,"H":66.0,"I":96.0,"L":40.0,"K":-56.0,"M":94.0,"F":41.0,"P":56.0,"S":120.0,"T":97.0,"W":18.0,"Y":41.0,"V":74.0}


###You can continuely add other properties of AADs to compute the descriptors of protein sequence.


_AAProperty=(_Hydrophobicity,_AvFlexibility,_Polarizability,_FreeEnergy,_ResidueASA,_ResidueVol,_Steric,_Mutability)

_AAPropertyName=('_Hydrophobicity','_AvFlexibility','_Polarizability','_FreeEnergy','_ResidueASA','_ResidueVol','_Steric','_Mutability')			 

##################################################################################################
def _mean(listvalue):
	"""
	The mean value of the list data.
	"""
	return sum(listvalue)/len(listvalue)
##################################################################################################
def _std(listvalue,ddof=1):
	"""
	The standard deviation of the list data.
	"""
	mean=_mean(listvalue)
	temp=[math.pow(i-mean,2) for i in listvalue]
	res=math.sqrt(sum(temp)/(len(listvalue)-ddof))
	return res
##################################################################################################

def NormalizeEachAAP(AAP):
	"""
	####################################################################################
	All of the amino acid indices are centralized and 
	
	standardized before the calculation.
	
	Usage:
	
	result=NormalizeEachAAP(AAP)
	
	Input: AAP is a dict form containing the properties of 20 amino acids.
	
	Output: result is the a dict form containing the normalized properties 
	
	of 20 amino acids.
	####################################################################################
	"""
	if len(AAP.values())!=20:
		print('You can not input the correct number of properities of Amino acids!')
	else:
		Result={}
		for i,j in AAP.items():
			Result[i]=(j-_mean(AAP.values()))/_std(AAP.values(),ddof=0)

	return Result

def CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,AAP,AAPName):
		
	"""
	####################################################################################
	you can use the function to compute MoreauBrotoAuto
	
	descriptors for different properties based on AADs.
	
	Usage:
	
	result=CalculateEachNormalizedMoreauBrotoAuto(protein,AAP,AAPName)
	
	Input: protein is a pure protein sequence.
	
	AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	
	AAPName is a string used for indicating the property (e.g., '_AvFlexibility'). 
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto autocorrelation
	
	descriptors based on the given property.
	####################################################################################
	"""
		
	AAPdic=NormalizeEachAAP(AAP)

	Result={}
	for i in range(1,31):
		temp=0
		for j in range(len(ProteinSequence)-i):
			temp=temp+AAPdic[ProteinSequence[j]]*AAPdic[ProteinSequence[j+1]]
		if len(ProteinSequence)-i==0:
			Result['MoreauBrotoAuto'+AAPName+str(i)]=round(temp/(len(ProteinSequence)),3)
		else:
			Result['MoreauBrotoAuto'+AAPName+str(i)]=round(temp/(len(ProteinSequence)-i),3)

	return Result


def CalculateEachMoranAuto(ProteinSequence,AAP,AAPName):

	"""
	####################################################################################
	you can use the function to compute MoranAuto
	
	descriptors for different properties based on AADs.
	
	Usage:
	
	result=CalculateEachMoranAuto(protein,AAP,AAPName)
	
	Input: protein is a pure protein sequence.
	
	AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	
	AAPName is a string used for indicating the property (e.g., '_AvFlexibility'). 
	
	Output: result is a dict form containing 30 Moran autocorrelation
	
	descriptors based on the given property.
	####################################################################################
	"""

	AAPdic=NormalizeEachAAP(AAP)

	cds=0
	for i in AALetter:
		cds=cds+(ProteinSequence.count(i))*(AAPdic[i])
	Pmean=cds/len(ProteinSequence)

	cc=[]
	for i in ProteinSequence:
		cc.append(AAPdic[i])

	K=(_std(cc,ddof=0))**2

	Result={}
	for i in range(1,31):
		temp=0
		for j in range(len(ProteinSequence)-i):
				
			temp=temp+(AAPdic[ProteinSequence[j]]-Pmean)*(AAPdic[ProteinSequence[j+i]]-Pmean)
		if len(ProteinSequence)-i==0:
			Result['MoranAuto'+AAPName+str(i)]=round(temp/(len(ProteinSequence))/K,3)
		else:
			Result['MoranAuto'+AAPName+str(i)]=round(temp/(len(ProteinSequence)-i)/K,3)

	return Result


def CalculateEachGearyAuto(ProteinSequence,AAP,AAPName):

	"""
	####################################################################################
	you can use the function to compute GearyAuto
	
	descriptors for different properties based on AADs.
	
	Usage:
	
	result=CalculateEachGearyAuto(protein,AAP,AAPName)
	
	Input: protein is a pure protein sequence.
	
	AAP is a dict form containing the properties of 20 amino acids (e.g., _AvFlexibility).
	
	AAPName is a string used for indicating the property (e.g., '_AvFlexibility'). 
	
	Output: result is a dict form containing 30 Geary autocorrelation
	
	descriptors based on the given property.
	####################################################################################
	"""

	AAPdic=NormalizeEachAAP(AAP)

	cc=[]
	for i in ProteinSequence:
		cc.append(AAPdic[i])

	K=((_std(cc))**2)*len(ProteinSequence)/(len(ProteinSequence)-1)
	Result={}
	for i in range(1,31):
		temp=0
		for j in range(len(ProteinSequence)-i):
				
			temp=temp+(AAPdic[ProteinSequence[j]]-AAPdic[ProteinSequence[j+i]])**2
		if len(ProteinSequence)-i==0:
			Result['GearyAuto'+AAPName+str(i)]=round(temp/(2*(len(ProteinSequence)))/K,3)
		else:
			Result['GearyAuto'+AAPName+str(i)]=round(temp/(2*(len(ProteinSequence)-i))/K,3)
	return Result


##################################################################################################

def CalculateNormalizedMoreauBrotoAuto(ProteinSequence,AAProperty,AAPropertyName): 

	"""
	####################################################################################
	A method used for computing MoreauBrotoAuto for all properties.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAuto(protein,AAP,AAPName)
	
	Input: protein is a pure protein sequence.
	
	AAProperty is a list or tuple form containing the properties of 20 amino acids (e.g., _AAProperty).
	
	AAPName is a list or tuple form used for indicating the property (e.g., '_AAPropertyName'). 
	
	Output: result is a dict form containing 30*p Normalized Moreau-Broto autocorrelation
	
	descriptors based on the given properties.
	####################################################################################
	
	"""
	Result={}
	for i in range(len(AAProperty)):
		Result[AAPropertyName[i]]=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,AAProperty[i],AAPropertyName[i])


	return Result


def CalculateMoranAuto(ProteinSequence,AAProperty,AAPropertyName):  
	"""
	####################################################################################
	A method used for computing MoranAuto for all properties
	
	Usage:
	
	result=CalculateMoranAuto(protein,AAP,AAPName)
	
	Input: protein is a pure protein sequence.
	
	AAProperty is a list or tuple form containing the properties of 20 amino acids (e.g., _AAProperty).
	
	AAPName is a list or tuple form used for indicating the property (e.g., '_AAPropertyName'). 
	
	Output: result is a dict form containing 30*p Moran autocorrelation
	
	descriptors based on the given properties.
	####################################################################################
	"""
	Result={}
	for i in range(len(AAProperty)):
		Result[AAPropertyName[i]]=CalculateEachMoranAuto(ProteinSequence,AAProperty[i],AAPropertyName[i])

	return Result



def CalculateGearyAuto(ProteinSequence,AAProperty,AAPropertyName):  
	"""
	####################################################################################
	A method used for computing GearyAuto for all properties
	
	Usage:
	
	result=CalculateGearyAuto(protein,AAP,AAPName)
	
	Input: protein is a pure protein sequence.
	
	AAProperty is a list or tuple form containing the properties of 20 amino acids (e.g., _AAProperty).
	
	AAPName is a list or tuple form used for indicating the property (e.g., '_AAPropertyName'). 
	
	Output: result is a dict form containing 30*p Geary autocorrelation
	
	descriptors based on the given properties.
	####################################################################################
	"""
	Result={}
	for i in range(len(AAProperty)):
		Result[AAPropertyName[i]]=CalculateEachGearyAuto(ProteinSequence,AAProperty[i],AAPropertyName[i])

	return Result


########################NormalizedMoreauBorto##################################
def CalculateNormalizedMoreauBrotoAutoHydrophobicity(ProteinSequence):

	"""
	####################################################################################
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on
	
	hydrophobicity.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoHydrophobicity(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation
	
	descriptors based on Hydrophobicity.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result


def CalculateNormalizedMoreauBrotoAutoAvFlexibility(ProteinSequence):

	"""
	####################################################################################
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on
	
	AvFlexibility.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoAvFlexibility(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation
	
	descriptors based on AvFlexibility.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_AvFlexibility,'_AvFlexibility')
	return result


def CalculateNormalizedMoreauBrotoAutoPolarizability(ProteinSequence):

	"""
	####################################################################################
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on
	
	Polarizability.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoPolarizability(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation
	
	descriptors based on Polarizability.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_Polarizability,'_Polarizability')
	return result


def CalculateNormalizedMoreauBrotoAutoFreeEnergy(ProteinSequence):

	"""
	####################################################################################
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on
	
	FreeEnergy.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoFreeEnergy(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation
	
	descriptors based on FreeEnergy.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_FreeEnergy,'_FreeEnergy')
	return result



def CalculateNormalizedMoreauBrotoAutoResidueASA(ProteinSequence):

	"""
	####################################################################################
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on
	
	ResidueASA.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoResidueASA(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation
	
	descriptors based on ResidueASA.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_ResidueASA,'_ResidueASA')
	return result


def CalculateNormalizedMoreauBrotoAutoResidueVol(ProteinSequence):

	"""
	####################################################################################
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on
	
	ResidueVol.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoResidueVol(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation
	
	descriptors based on ResidueVol.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_ResidueVol,'_ResidueVol')
	return result
	
def CalculateNormalizedMoreauBrotoAutoSteric(ProteinSequence):

	"""
	####################################################################################
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on Steric.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoSteric(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation
	
	descriptors based on Steric.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_Steric,'_Steric')
	return result


def CalculateNormalizedMoreauBrotoAutoMutability(ProteinSequence):

	"""
	####################################################################################
	Calculte the NormalizedMoreauBorto Autocorrelation descriptors based on Mutability.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoMutability(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Normalized Moreau-Broto Autocorrelation
	
	descriptors based on Mutability.
	####################################################################################
	"""
	
	result=CalculateEachNormalizedMoreauBrotoAuto(ProteinSequence,_Mutability,'_Mutability')
	return result
############################################################################

##############################MoranAuto######################################
def CalculateMoranAutoHydrophobicity(ProteinSequence):

	"""
	####################################################################################
	Calculte the MoranAuto Autocorrelation descriptors based on hydrophobicity.
	
	Usage:
	
	result=CalculateMoranAutoHydrophobicity(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Moran Autocorrelation
	
	descriptors based on hydrophobicity.
	####################################################################################
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result
	

def CalculateMoranAutoAvFlexibility(ProteinSequence):

	"""
	####################################################################################
	Calculte the MoranAuto Autocorrelation descriptors based on
	
	AvFlexibility.
	
	Usage:
	
	result=CalculateMoranAutoAvFlexibility(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Moran Autocorrelation
	
	descriptors based on AvFlexibility.
	####################################################################################
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_AvFlexibility,'_AvFlexibility')
	return result


def CalculateMoranAutoPolarizability(ProteinSequence):

	"""
	####################################################################################
	Calculte the MoranAuto Autocorrelation descriptors based on
	
	Polarizability.
	
	Usage:
	
	result=CalculateMoranAutoPolarizability(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Moran Autocorrelation
	
	descriptors based on Polarizability.
	####################################################################################
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_Polarizability,'_Polarizability')
	return result


def CalculateMoranAutoFreeEnergy(ProteinSequence):

	"""
	####################################################################################
	Calculte the MoranAuto Autocorrelation descriptors based on
	
	FreeEnergy.
	
	Usage:
	
	result=CalculateMoranAutoFreeEnergy(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Moran Autocorrelation
	
	descriptors based on FreeEnergy.
	####################################################################################
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_FreeEnergy,'_FreeEnergy')
	return result



def CalculateMoranAutoResidueASA(ProteinSequence):

	"""
	####################################################################################
	Calculte the MoranAuto Autocorrelation descriptors based on
	
	ResidueASA.
	
	Usage:
	
	result=CalculateMoranAutoResidueASA(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Moran Autocorrelation
	
	descriptors based on ResidueASA.
	####################################################################################
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_ResidueASA,'_ResidueASA')
	return result


def CalculateMoranAutoResidueVol(ProteinSequence):

	"""
	####################################################################################
	Calculte the MoranAuto Autocorrelation descriptors based on
	
	ResidueVol.
	
	Usage:
	
	result=CalculateMoranAutoResidueVol(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Moran Autocorrelation
	
	descriptors based on ResidueVol.
	####################################################################################
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_ResidueVol,'_ResidueVol')
	return result
	
def CalculateMoranAutoSteric(ProteinSequence):

	"""
	####################################################################################
	Calculte the MoranAuto Autocorrelation descriptors based on
	
	AutoSteric.
	
	Usage:
	
	result=CalculateMoranAutoSteric(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Moran Autocorrelation
	
	descriptors based on AutoSteric.
	####################################################################################
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_Steric,'_Steric')
	return result


def CalculateMoranAutoMutability(ProteinSequence):

	"""
	####################################################################################
	Calculte the MoranAuto Autocorrelation descriptors based on
	
	Mutability.
	
	Usage:
	
	result=CalculateMoranAutoMutability(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Moran Autocorrelation
	
	descriptors based on Mutability.
	####################################################################################
	"""
	
	result=CalculateEachMoranAuto(ProteinSequence,_Mutability,'_Mutability')
	return result
############################################################################


################################GearyAuto#####################################
def CalculateGearyAutoHydrophobicity(ProteinSequence):

	"""
	####################################################################################
	Calculte the GearyAuto Autocorrelation descriptors based on
	
	hydrophobicity.
	
	Usage:
	
	result=CalculateGearyAutoHydrophobicity(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Geary Autocorrelation
	
	descriptors based on hydrophobicity.
	####################################################################################
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_Hydrophobicity,'_Hydrophobicity')
	return result
	

def CalculateGearyAutoAvFlexibility(ProteinSequence):

	"""
	####################################################################################
	Calculte the GearyAuto Autocorrelation descriptors based on
	
	AvFlexibility.
	
	Usage:
	result=CalculateGearyAutoAvFlexibility(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Geary Autocorrelation
	
	descriptors based on AvFlexibility.
	####################################################################################
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_AvFlexibility,'_AvFlexibility')
	return result


def CalculateGearyAutoPolarizability(ProteinSequence):

	"""
	####################################################################################
	Calculte the GearyAuto Autocorrelation descriptors based on
	
	Polarizability.
	
	Usage:
	
	result=CalculateGearyAutoPolarizability(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Geary Autocorrelation
	
	descriptors based on Polarizability.
	####################################################################################
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_Polarizability,'_Polarizability')
	return result


def CalculateGearyAutoFreeEnergy(ProteinSequence):
	
	"""
	####################################################################################
	Calculte the GearyAuto Autocorrelation descriptors based on
	
	FreeEnergy.
	
	Usage:
	
	result=CalculateGearyAutoFreeEnergy(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Geary Autocorrelation
	
	descriptors based on FreeEnergy.
	####################################################################################
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_FreeEnergy,'_FreeEnergy')
	return result



def CalculateGearyAutoResidueASA(ProteinSequence):
	
	"""
	####################################################################################
	Calculte the GearyAuto Autocorrelation descriptors based on
	
	ResidueASA.
	
	Usage:
	
	result=CalculateGearyAutoResidueASA(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Geary Autocorrelation
	
	descriptors based on ResidueASA.
	####################################################################################
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_ResidueASA,'_ResidueASA')
	return result


def CalculateGearyAutoResidueVol(ProteinSequence):
	
	"""
	####################################################################################
	Calculte the GearyAuto Autocorrelation descriptors based on
	
	ResidueVol.
	
	Usage:
	
	result=CalculateGearyAutoResidueVol(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Geary Autocorrelation
	
	descriptors based on ResidueVol.
	####################################################################################
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_ResidueVol,'_ResidueVol')
	return result
	
def CalculateGearyAutoSteric(ProteinSequence):
	
	"""
	####################################################################################
	Calculte the GearyAuto Autocorrelation descriptors based on
	
	Steric.
	
	Usage:
	
	result=CalculateGearyAutoSteric(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Geary Autocorrelation
	
	descriptors based on Steric.
	####################################################################################
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_Steric,'_Steric')
	return result


def CalculateGearyAutoMutability(ProteinSequence):

	"""
	####################################################################################
	Calculte the GearyAuto Autocorrelation descriptors based on
	
	Mutability.
	
	Usage:
	
	result=CalculateGearyAutoMutability(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30 Geary Autocorrelation
	
	descriptors based on Mutability.
	####################################################################################
	"""
	
	result=CalculateEachGearyAuto(ProteinSequence,_Mutability,'_Mutability')
	return result
##################################################################################################

def CalculateNormalizedMoreauBrotoAutoTotal(ProteinSequence):
	"""
	####################################################################################
	A method used for computing normalized Moreau Broto autocorrelation descriptors based 
	
	on 8 proterties of AADs.
	
	Usage:
	
	result=CalculateNormalizedMoreauBrotoAutoTotal(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30*8=240 normalized Moreau Broto 
	
	autocorrelation descriptors based on the given properties(i.e., _AAPropert).
	#################################################################################### 
	"""
	result={}
	result.update(CalculateNormalizedMoreauBrotoAutoHydrophobicity(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoAvFlexibility(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoPolarizability(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoFreeEnergy(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoResidueASA(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoResidueVol(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoSteric(ProteinSequence))
	result.update(CalculateNormalizedMoreauBrotoAutoMutability(ProteinSequence))
	return result

def CalculateMoranAutoTotal(ProteinSequence):
	"""
	####################################################################################
	A method used for computing Moran autocorrelation descriptors based on 8 properties of AADs.
	
	Usage:
	
	result=CalculateMoranAutoTotal(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30*8=240 Moran
	
	autocorrelation descriptors based on the given properties(i.e., _AAPropert).
	####################################################################################
	"""
	result={}
	result.update(CalculateMoranAutoHydrophobicity(ProteinSequence))
	result.update(CalculateMoranAutoAvFlexibility(ProteinSequence))
	result.update(CalculateMoranAutoPolarizability(ProteinSequence))
	result.update(CalculateMoranAutoFreeEnergy(ProteinSequence))
	result.update(CalculateMoranAutoResidueASA(ProteinSequence))
	result.update(CalculateMoranAutoResidueVol(ProteinSequence))
	result.update(CalculateMoranAutoSteric(ProteinSequence))
	result.update(CalculateMoranAutoMutability(ProteinSequence))
	return result

def CalculateGearyAutoTotal(ProteinSequence):
	"""
	####################################################################################
	A method used for computing Geary autocorrelation descriptors based on 8 properties of AADs.
	
	Usage:
	
	result=CalculateGearyAutoTotal(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30*8=240 Geary
	
	autocorrelation descriptors based on the given properties(i.e., _AAPropert).
	####################################################################################
	"""
	result={}
	result.update(CalculateGearyAutoHydrophobicity(ProteinSequence))
	result.update(CalculateGearyAutoAvFlexibility(ProteinSequence))
	result.update(CalculateGearyAutoPolarizability(ProteinSequence))
	result.update(CalculateGearyAutoFreeEnergy(ProteinSequence))
	result.update(CalculateGearyAutoResidueASA(ProteinSequence))
	result.update(CalculateGearyAutoResidueVol(ProteinSequence))
	result.update(CalculateGearyAutoSteric(ProteinSequence))
	result.update(CalculateGearyAutoMutability(ProteinSequence))
	return result

##################################################################################################
def CalculateAutoTotal(ProteinSequence):
	"""
	####################################################################################
	A method used for computing all autocorrelation descriptors based on 8 properties of AADs.
	
	Usage:
	
	result=CalculateGearyAutoTotal(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing 30*8*3=720 normalized Moreau Broto, Moran, and Geary
	
	autocorrelation descriptors based on the given properties(i.e., _AAPropert).
	####################################################################################
	"""
	result={}
	result.update(CalculateNormalizedMoreauBrotoAutoTotal(ProteinSequence))
	result.update(CalculateMoranAutoTotal(ProteinSequence))
	result.update(CalculateGearyAutoTotal(ProteinSequence))
	return result

##################################################################################################
if __name__=="__main__":
    
    ps="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"
    res1=CalculateNormalizedMoreauBrotoAutoTotal(ps)
    res2=CalculateMoranAutoTotal(ps)
    res3=CalculateGearyAutoTotal(ps)
    print(len(res1), len(res2), len(res3))
    