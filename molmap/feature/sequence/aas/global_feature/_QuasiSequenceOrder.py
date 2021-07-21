# -*- coding: utf-8 -*-
"""
##############################################################################################
This module is used for computing the quasi sequence order descriptors based on the 

given protein sequence. We can obtain two types of descriptors: Sequence-order-coupling

number and quasi-sequence-order descriptors. Two distance matrixes between 20 amino acids

are employed. You can freely use and distribute it. If you have any problem, please contact

us immediately.

References:

[1]:Kuo-Chen Chou. Prediction of Protein Subcellar Locations by Incorporating

Quasi-Sequence-Order Effect. Biochemical and Biophysical Research Communications 

2000, 278, 477-483.

[2]: Kuo-Chen Chou and Yu-Dong Cai. Prediction of Protein sucellular locations by

GO-FunD-PseAA predictor, Biochemical and Biophysical Research Communications,

2004, 320, 1236-1239.

[3]:Gisbert Schneider and Paul wrede. The Rational Design of Amino Acid

Sequences by Artifical Neural Networks and Simulated Molecular Evolution: Do

Novo Design of an Idealized Leader Cleavge Site. Biophys Journal, 1994, 66,

335-344.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2012.09.03

Email: oriental-cds@163.com

##############################################################################################
"""

import math


AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
## Distance is the Schneider-Wrede physicochemical distance matrix used by Chou et. al. 
_Distance1={"GW":0.923, "GV":0.464, "GT":0.272, "GS":0.158, "GR":1.0, "GQ":0.467, "GP":0.323, "GY":0.728, "GG":0.0, "GF":0.727, "GE":0.807, "GD":0.776, "GC":0.312, "GA":0.206, "GN":0.381, "GM":0.557, "GL":0.591, "GK":0.894, "GI":0.592, "GH":0.769, "ME":0.879, "MD":0.932, "MG":0.569, "MF":0.182, "MA":0.383, "MC":0.276, "MM":0.0, "ML":0.062, "MN":0.447, "MI":0.058, "MH":0.648, "MK":0.884, "MT":0.358, "MW":0.391, "MV":0.12, "MQ":0.372, "MP":0.285, "MS":0.417, "MR":1.0, "MY":0.255, "FP":0.42, "FQ":0.459, "FR":1.0, "FS":0.548, "FT":0.499, "FV":0.252, "FW":0.207, "FY":0.179, "FA":0.508, "FC":0.405, "FD":0.977, "FE":0.918, "FF":0.0, "FG":0.69, "FH":0.663, "FI":0.128, "FK":0.903, "FL":0.131, "FM":0.169, "FN":0.541, "SY":0.615, "SS":0.0, "SR":1.0, "SQ":0.358, "SP":0.181, "SW":0.827, "SV":0.342, "ST":0.174, "SK":0.883, "SI":0.478, "SH":0.718, "SN":0.289, "SM":0.44, "SL":0.474, "SC":0.185, "SA":0.1, "SG":0.17, "SF":0.622, "SE":0.812, "SD":0.801, "YI":0.23, "YH":0.678, "YK":0.904, "YM":0.268, "YL":0.219, "YN":0.512, "YA":0.587, "YC":0.478, "YE":0.932, "YD":1.0, "YG":0.782, "YF":0.202, "YY":0.0, "YQ":0.404, "YP":0.444, "YS":0.612, "YR":0.995, "YT":0.557, "YW":0.244, "YV":0.328, "LF":0.139, "LG":0.596, "LD":0.944, "LE":0.892, "LC":0.296, "LA":0.405, "LN":0.452, "LL":0.0, "LM":0.062, "LK":0.893, "LH":0.653, "LI":0.013, "LV":0.133, "LW":0.341, "LT":0.397, "LR":1.0, "LS":0.443, "LP":0.309, "LQ":0.376, "LY":0.205, "RT":0.808, "RV":0.914, "RW":1.0, "RP":0.796, "RQ":0.668, "RR":0.0, "RS":0.86, "RY":0.859, "RD":0.305, "RE":0.225, "RF":0.977, "RG":0.928, "RA":0.919, "RC":0.905, "RL":0.92, "RM":0.908, "RN":0.69, "RH":0.498, "RI":0.929, "RK":0.141, "VH":0.649, "VI":0.135, "EM":0.83, "EL":0.854, "EN":0.599, "EI":0.86, "EH":0.406, "EK":0.143, "EE":0.0, "ED":0.133, "EG":0.779, "EF":0.932, "EA":0.79, "EC":0.788, "VM":0.12, "EY":0.837, "VN":0.38, "ET":0.682, "EW":1.0, "EV":0.824, "EQ":0.598, "EP":0.688, "ES":0.726, "ER":0.234, "VP":0.212, "VQ":0.339, "VR":1.0, "VT":0.305, "VW":0.472, "KC":0.871, "KA":0.889, "KG":0.9, "KF":0.957, "KE":0.149, "KD":0.279, "KK":0.0, "KI":0.899, "KH":0.438, "KN":0.667, "KM":0.871, "KL":0.892, "KS":0.825, "KR":0.154, "KQ":0.639, "KP":0.757, "KW":1.0, "KV":0.882, "KT":0.759, "KY":0.848, "DN":0.56, "DL":0.841, "DM":0.819, "DK":0.249, "DH":0.435, "DI":0.847, "DF":0.924, "DG":0.697, "DD":0.0, "DE":0.124, "DC":0.742, "DA":0.729, "DY":0.836, "DV":0.797, "DW":1.0, "DT":0.649, "DR":0.295, "DS":0.667, "DP":0.657, "DQ":0.584, "QQ":0.0, "QP":0.272, "QS":0.461, "QR":1.0, "QT":0.389, "QW":0.831, "QV":0.464, "QY":0.522, "QA":0.512, "QC":0.462, "QE":0.861, "QD":0.903, "QG":0.648, "QF":0.671, "QI":0.532, "QH":0.765, "QK":0.881, "QM":0.505, "QL":0.518, "QN":0.181, "WG":0.829, "WF":0.196, "WE":0.931, "WD":1.0, "WC":0.56, "WA":0.658, "WN":0.631, "WM":0.344, "WL":0.304, "WK":0.892, "WI":0.305, "WH":0.678, "WW":0.0, "WV":0.418, "WT":0.638, "WS":0.689, "WR":0.968, "WQ":0.538, "WP":0.555, "WY":0.204, "PR":1.0, "PS":0.196, "PP":0.0, "PQ":0.228, "PV":0.244, "PW":0.72, "PT":0.161, "PY":0.481, "PC":0.179, "PA":0.22, "PF":0.515, "PG":0.376, "PD":0.852, "PE":0.831, "PK":0.875, "PH":0.696, "PI":0.363, "PN":0.231, "PL":0.357, "PM":0.326, "CK":0.887, "CI":0.304, "CH":0.66, "CN":0.324, "CM":0.277, "CL":0.301, "CC":0.0, "CA":0.114, "CG":0.32, "CF":0.437, "CE":0.838, "CD":0.847, "CY":0.457, "CS":0.176, "CR":1.0, "CQ":0.341, "CP":0.157, "CW":0.639, "CV":0.167, "CT":0.233, "IY":0.213, "VA":0.275, "VC":0.165, "VD":0.9, "VE":0.867, "VF":0.269, "VG":0.471, "IQ":0.383, "IP":0.311, "IS":0.443, "IR":1.0, "VL":0.134, "IT":0.396, "IW":0.339, "IV":0.133, "II":0.0, "IH":0.652, "IK":0.892, "VS":0.322, "IM":0.057, "IL":0.013, "VV":0.0, "IN":0.457, "IA":0.403, "VY":0.31, "IC":0.296, "IE":0.891, "ID":0.942, "IG":0.592, "IF":0.134, "HY":0.821, "HR":0.697, "HS":0.865, "HP":0.777, "HQ":0.716, "HV":0.831, "HW":0.981, "HT":0.834, "HK":0.566, "HH":0.0, "HI":0.848, "HN":0.754, "HL":0.842, "HM":0.825, "HC":0.836, "HA":0.896, "HF":0.907, "HG":1.0, "HD":0.629, "HE":0.547, "NH":0.78, "NI":0.615, "NK":0.891, "NL":0.603, "NM":0.588, "NN":0.0, "NA":0.424, "NC":0.425, "ND":0.838, "NE":0.835, "NF":0.766, "NG":0.512, "NY":0.641, "NP":0.266, "NQ":0.175, "NR":1.0, "NS":0.361, "NT":0.368, "NV":0.503, "NW":0.945, "TY":0.596, "TV":0.345, "TW":0.816, "TT":0.0, "TR":1.0, "TS":0.185, "TP":0.159, "TQ":0.322, "TN":0.315, "TL":0.453, "TM":0.403, "TK":0.866, "TH":0.737, "TI":0.455, "TF":0.604, "TG":0.312, "TD":0.83, "TE":0.812, "TC":0.261, "TA":0.251, "AA":0.0, "AC":0.112, "AE":0.827, "AD":0.819, "AG":0.208, "AF":0.54, "AI":0.407, "AH":0.696, "AK":0.891, "AM":0.379, "AL":0.406, "AN":0.318, "AQ":0.372, "AP":0.191, "AS":0.094, "AR":1.0, "AT":0.22, "AW":0.739, "AV":0.273, "AY":0.552, "VK":0.889 }

## Distance is the Grantham chemical distance matrix used by Grantham et. al. 
_Distance2={"GW":0.923, "GV":0.464, "GT":0.272, "GS":0.158, "GR":1.0, "GQ":0.467, "GP":0.323, "GY":0.728, "GG":0.0, "GF":0.727, "GE":0.807, "GD":0.776, "GC":0.312, "GA":0.206, "GN":0.381, "GM":0.557, "GL":0.591, "GK":0.894, "GI":0.592, "GH":0.769, "ME":0.879, "MD":0.932, "MG":0.569, "MF":0.182, "MA":0.383, "MC":0.276, "MM":0.0, "ML":0.062, "MN":0.447, "MI":0.058, "MH":0.648, "MK":0.884, "MT":0.358, "MW":0.391, "MV":0.12, "MQ":0.372, "MP":0.285, "MS":0.417, "MR":1.0, "MY":0.255, "FP":0.42, "FQ":0.459, "FR":1.0, "FS":0.548, "FT":0.499, "FV":0.252, "FW":0.207, "FY":0.179, "FA":0.508, "FC":0.405, "FD":0.977, "FE":0.918, "FF":0.0, "FG":0.69, "FH":0.663, "FI":0.128, "FK":0.903, "FL":0.131, "FM":0.169, "FN":0.541, "SY":0.615, "SS":0.0, "SR":1.0, "SQ":0.358, "SP":0.181, "SW":0.827, "SV":0.342, "ST":0.174, "SK":0.883, "SI":0.478, "SH":0.718, "SN":0.289, "SM":0.44, "SL":0.474, "SC":0.185, "SA":0.1, "SG":0.17, "SF":0.622, "SE":0.812, "SD":0.801, "YI":0.23, "YH":0.678, "YK":0.904, "YM":0.268, "YL":0.219, "YN":0.512, "YA":0.587, "YC":0.478, "YE":0.932, "YD":1.0, "YG":0.782, "YF":0.202, "YY":0.0, "YQ":0.404, "YP":0.444, "YS":0.612, "YR":0.995, "YT":0.557, "YW":0.244, "YV":0.328, "LF":0.139, "LG":0.596, "LD":0.944, "LE":0.892, "LC":0.296, "LA":0.405, "LN":0.452, "LL":0.0, "LM":0.062, "LK":0.893, "LH":0.653, "LI":0.013, "LV":0.133, "LW":0.341, "LT":0.397, "LR":1.0, "LS":0.443, "LP":0.309, "LQ":0.376, "LY":0.205, "RT":0.808, "RV":0.914, "RW":1.0, "RP":0.796, "RQ":0.668, "RR":0.0, "RS":0.86, "RY":0.859, "RD":0.305, "RE":0.225, "RF":0.977, "RG":0.928, "RA":0.919, "RC":0.905, "RL":0.92, "RM":0.908, "RN":0.69, "RH":0.498, "RI":0.929, "RK":0.141, "VH":0.649, "VI":0.135, "EM":0.83, "EL":0.854, "EN":0.599, "EI":0.86, "EH":0.406, "EK":0.143, "EE":0.0, "ED":0.133, "EG":0.779, "EF":0.932, "EA":0.79, "EC":0.788, "VM":0.12, "EY":0.837, "VN":0.38, "ET":0.682, "EW":1.0, "EV":0.824, "EQ":0.598, "EP":0.688, "ES":0.726, "ER":0.234, "VP":0.212, "VQ":0.339, "VR":1.0, "VT":0.305, "VW":0.472, "KC":0.871, "KA":0.889, "KG":0.9, "KF":0.957, "KE":0.149, "KD":0.279, "KK":0.0, "KI":0.899, "KH":0.438, "KN":0.667, "KM":0.871, "KL":0.892, "KS":0.825, "KR":0.154, "KQ":0.639, "KP":0.757, "KW":1.0, "KV":0.882, "KT":0.759, "KY":0.848, "DN":0.56, "DL":0.841, "DM":0.819, "DK":0.249, "DH":0.435, "DI":0.847, "DF":0.924, "DG":0.697, "DD":0.0, "DE":0.124, "DC":0.742, "DA":0.729, "DY":0.836, "DV":0.797, "DW":1.0, "DT":0.649, "DR":0.295, "DS":0.667, "DP":0.657, "DQ":0.584, "QQ":0.0, "QP":0.272, "QS":0.461, "QR":1.0, "QT":0.389, "QW":0.831, "QV":0.464, "QY":0.522, "QA":0.512, "QC":0.462, "QE":0.861, "QD":0.903, "QG":0.648, "QF":0.671, "QI":0.532, "QH":0.765, "QK":0.881, "QM":0.505, "QL":0.518, "QN":0.181, "WG":0.829, "WF":0.196, "WE":0.931, "WD":1.0, "WC":0.56, "WA":0.658, "WN":0.631, "WM":0.344, "WL":0.304, "WK":0.892, "WI":0.305, "WH":0.678, "WW":0.0, "WV":0.418, "WT":0.638, "WS":0.689, "WR":0.968, "WQ":0.538, "WP":0.555, "WY":0.204, "PR":1.0, "PS":0.196, "PP":0.0, "PQ":0.228, "PV":0.244, "PW":0.72, "PT":0.161, "PY":0.481, "PC":0.179, "PA":0.22, "PF":0.515, "PG":0.376, "PD":0.852, "PE":0.831, "PK":0.875, "PH":0.696, "PI":0.363, "PN":0.231, "PL":0.357, "PM":0.326, "CK":0.887, "CI":0.304, "CH":0.66, "CN":0.324, "CM":0.277, "CL":0.301, "CC":0.0, "CA":0.114, "CG":0.32, "CF":0.437, "CE":0.838, "CD":0.847, "CY":0.457, "CS":0.176, "CR":1.0, "CQ":0.341, "CP":0.157, "CW":0.639, "CV":0.167, "CT":0.233, "IY":0.213, "VA":0.275, "VC":0.165, "VD":0.9, "VE":0.867, "VF":0.269, "VG":0.471, "IQ":0.383, "IP":0.311, "IS":0.443, "IR":1.0, "VL":0.134, "IT":0.396, "IW":0.339, "IV":0.133, "II":0.0, "IH":0.652, "IK":0.892, "VS":0.322, "IM":0.057, "IL":0.013, "VV":0.0, "IN":0.457, "IA":0.403, "VY":0.31, "IC":0.296, "IE":0.891, "ID":0.942, "IG":0.592, "IF":0.134, "HY":0.821, "HR":0.697, "HS":0.865, "HP":0.777, "HQ":0.716, "HV":0.831, "HW":0.981, "HT":0.834, "HK":0.566, "HH":0.0, "HI":0.848, "HN":0.754, "HL":0.842, "HM":0.825, "HC":0.836, "HA":0.896, "HF":0.907, "HG":1.0, "HD":0.629, "HE":0.547, "NH":0.78, "NI":0.615, "NK":0.891, "NL":0.603, "NM":0.588, "NN":0.0, "NA":0.424, "NC":0.425, "ND":0.838, "NE":0.835, "NF":0.766, "NG":0.512, "NY":0.641, "NP":0.266, "NQ":0.175, "NR":1.0, "NS":0.361, "NT":0.368, "NV":0.503, "NW":0.945, "TY":0.596, "TV":0.345, "TW":0.816, "TT":0.0, "TR":1.0, "TS":0.185, "TP":0.159, "TQ":0.322, "TN":0.315, "TL":0.453, "TM":0.403, "TK":0.866, "TH":0.737, "TI":0.455, "TF":0.604, "TG":0.312, "TD":0.83, "TE":0.812, "TC":0.261, "TA":0.251, "AA":0.0, "AC":0.112, "AE":0.827, "AD":0.819, "AG":0.208, "AF":0.54, "AI":0.407, "AH":0.696, "AK":0.891, "AM":0.379, "AL":0.406, "AN":0.318, "AQ":0.372, "AP":0.191, "AS":0.094, "AR":1.0, "AT":0.22, "AW":0.739, "AV":0.273, "AY":0.552, "VK":0.889 }
#############################################################################################
#############################################################################################
def GetSequenceOrderCouplingNumber(ProteinSequence,d=1,distancematrix=_Distance1):
	"""
	###############################################################################
	Computing the dth-rank sequence order coupling number for a protein.
	
	Usage:
	
	result = GetSequenceOrderCouplingNumber(protein,d)
	
	Input: protein is a pure protein sequence.
	
	d is the gap between two amino acids.
	
	Output: result is numeric value.
	
	###############################################################################
	"""
	NumProtein=len(ProteinSequence)
	tau=0.0
	for i in range(NumProtein-d):
		temp1=ProteinSequence[i]
		temp2=ProteinSequence[i+d]
		tau=tau+math.pow(distancematrix[temp1+temp2],2)
	return round(tau,3)
#############################################################################################
def GetSequenceOrderCouplingNumberp(ProteinSequence,maxlag=30,distancematrix=_Distance1):
	"""
	###############################################################################
	Computing the sequence order coupling numbers from 1 to maxlag
	
	for a given protein sequence based on the user-defined property.
	
	Usage:
	
	result = GetSequenceOrderCouplingNumberp(protein, maxlag,distancematrix)
	
	Input: protein is a pure protein sequence
	
	maxlag is the maximum lag and the length of the protein should be larger
	
	than maxlag. default is 30.
	
	distancematrix is the a dict form containing 400 distance values
	
	Output: result is a dict form containing all sequence order coupling numbers based
	
	on the given property
	###############################################################################
	"""
	Tau={}
	for i in range(maxlag):
		Tau["tau"+str(i+1)]=GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	return Tau
#############################################################################################
def GetSequenceOrderCouplingNumberSW(ProteinSequence,maxlag=30,distancematrix=_Distance1):
	"""
	###############################################################################
	Computing the sequence order coupling numbers from 1 to maxlag
	
	for a given protein sequence based on the Schneider-Wrede physicochemical 
	
	distance matrix
	
	Usage:
	
	result = GetSequenceOrderCouplingNumberSW(protein, maxlag,distancematrix)
	
	Input: protein is a pure protein sequence
	
	maxlag is the maximum lag and the length of the protein should be larger
	
	than maxlag. default is 30.
	
	distancematrix is a dict form containing Schneider-Wrede physicochemical 
	
	distance matrix. omitted!
	
	Output: result is a dict form containing all sequence order coupling numbers based
	
	on the Schneider-Wrede physicochemical distance matrix
	###############################################################################
	"""
	Tau={}
	for i in range(maxlag):
		Tau["tausw"+str(i+1)]=GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	return Tau

#############################################################################################
def GetSequenceOrderCouplingNumberGrant(ProteinSequence,maxlag=30,distancematrix=_Distance2):
	"""

	###############################################################################
	Computing the sequence order coupling numbers from 1 to maxlag

	for a given protein sequence based on the Grantham chemical distance
	
	matrix. 

	Usage:

	result = GetSequenceOrderCouplingNumberGrant(protein, maxlag,distancematrix)
	
	Input: protein is a pure protein sequence
	
	maxlag is the maximum lag and the length of the protein should be larger
	
	than maxlag. default is 30.
	
	distancematrix is a dict form containing Grantham chemical distance
	
	matrix. omitted!
	
	Output: result is a dict form containing all sequence order coupling numbers
	
	based on the Grantham chemical distance matrix
	###############################################################################
	"""
	Tau={}
	for i in range(maxlag):
		Tau["taugrant"+str(i+1)]=GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	return Tau
#############################################################################################
def GetSequenceOrderCouplingNumberTotal(ProteinSequence,maxlag=30):
	"""
	###############################################################################
	Computing the sequence order coupling numbers from 1 to maxlag
	for a given protein sequence.
	Usage:
	result = GetSequenceOrderCouplingNumberTotal(protein, maxlag)
	
	Input: protein is a pure protein sequence
	
	maxlag is the maximum lag and the length of the protein should be larger
	
	than maxlag. default is 30.
	
	Output: result is a dict form containing all sequence order coupling numbers
	###############################################################################
	"""
	Tau={}
	Tau.update(GetSequenceOrderCouplingNumberSW(ProteinSequence,maxlag=maxlag))
	Tau.update(GetSequenceOrderCouplingNumberGrant(ProteinSequence,maxlag=maxlag))
	return Tau
#############################################################################################
def GetAAComposition(ProteinSequence):

	"""
	###############################################################################
	Calculate the composition of Amino acids 
	
	for a given protein sequence.
	
	Usage:
	
	result=CalculateAAComposition(protein)
	
	Input: protein is a pure protein sequence.
	
	Output: result is a dict form containing the composition of 
	
	20 amino acids.
	###############################################################################
	"""
	LengthSequence=len(ProteinSequence)
	Result={}
	for i in AALetter:
		Result[i]=round(float(ProteinSequence.count(i))/LengthSequence,3)
	return Result

#############################################################################################
def GetQuasiSequenceOrder1(ProteinSequence,maxlag=30,weight=0.1,distancematrix={}):
	"""
	###############################################################################
	Computing the first 20 quasi-sequence-order descriptors for 
	
	a given protein sequence.
	
	Usage:
	
	result = GetQuasiSequenceOrder1(protein,maxlag,weigt)
	
	see method GetQuasiSequenceOrder for the choice of parameters.
	###############################################################################
	"""
	rightpart=0.0
	for i in range(maxlag):
		rightpart=rightpart+GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['QSO'+str(index+1)]=round(AAC[i]/temp,6)
	
	return result


#############################################################################################
def GetQuasiSequenceOrder2(ProteinSequence,maxlag=30,weight=0.1,distancematrix={}):
	"""
	###############################################################################
	Computing the last maxlag quasi-sequence-order descriptors for 
	
	a given protein sequence.
	
	Usage:
	
	result = GetQuasiSequenceOrder2(protein,maxlag,weigt)
	
	see method GetQuasiSequenceOrder for the choice of parameters.
	###############################################################################
	"""
	rightpart=[]
	for i in range(maxlag):
		rightpart.append(GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix))

	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+maxlag):
		result['QSO'+str(index+1)]=round(weight*rightpart[index-20]/temp,6)
	
	return result


#############################################################################################
def GetQuasiSequenceOrder1SW(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance1):
	"""
	###############################################################################
	Computing the first 20 quasi-sequence-order descriptors for 
	
	a given protein sequence.
	
	Usage:
	
	result = GetQuasiSequenceOrder1SW(protein,maxlag,weigt)
	
	see method GetQuasiSequenceOrder for the choice of parameters.
	###############################################################################
	"""
	rightpart=0.0
	for i in range(maxlag):
		rightpart=rightpart+GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['QSOSW'+str(index+1)]=round(AAC[i]/temp,6)
	
	return result


#############################################################################################
def GetQuasiSequenceOrder2SW(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance1):
	"""
	###############################################################################
	Computing the last maxlag quasi-sequence-order descriptors for 
	

	a given protein sequence.
	
	Usage:
	
	result = GetQuasiSequenceOrder2SW(protein,maxlag,weigt)
	
	see method GetQuasiSequenceOrder for the choice of parameters.
	###############################################################################

	"""
	rightpart=[]
	for i in range(maxlag):
		rightpart.append(GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix))

	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+maxlag):
		result['QSOSW'+str(index+1)]=round(weight*rightpart[index-20]/temp,6)
	
	return result
#############################################################################################
def GetQuasiSequenceOrder1Grant(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance2):
	"""
	###############################################################################
	Computing the first 20 quasi-sequence-order descriptors for 
	
	a given protein sequence.
	
	Usage:
	
	result = GetQuasiSequenceOrder1Grant(protein,maxlag,weigt)
	
	see method GetQuasiSequenceOrder for the choice of parameters.
	###############################################################################
	"""
	rightpart=0.0
	for i in range(maxlag):
		rightpart=rightpart+GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix)
	AAC=GetAAComposition(ProteinSequence)
	result={}
	temp=1+weight*rightpart
	for index,i in enumerate(AALetter):
		result['QSOgrant'+str(index+1)]=round(AAC[i]/temp,6)
	
	return result


#############################################################################################
def GetQuasiSequenceOrder2Grant(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance2):
	"""
	###############################################################################
	Computing the last maxlag quasi-sequence-order descriptors for 
	

	a given protein sequence.
	
	Usage:
	
	result = GetQuasiSequenceOrder2Grant(protein,maxlag,weigt)
	
	see method GetQuasiSequenceOrder for the choice of parameters.
	###############################################################################

	"""
	rightpart=[]
	for i in range(maxlag):
		rightpart.append(GetSequenceOrderCouplingNumber(ProteinSequence,i+1,distancematrix))

	result={}
	temp=1+weight*sum(rightpart)
	for index in range(20,20+maxlag):
		result['QSOgrant'+str(index+1)]=round(weight*rightpart[index-20]/temp,6)
	
	return result
#############################################################################################
def GetQuasiSequenceOrder(ProteinSequence,maxlag=30,weight=0.1):
	"""
	###############################################################################
	Computing quasi-sequence-order descriptors for a given protein.
	
	[1]:Kuo-Chen Chou. Prediction of Protein Subcellar Locations by 
	
	Incorporating Quasi-Sequence-Order Effect. Biochemical and Biophysical
	
	Research Communications 2000, 278, 477-483.
	
	Usage:
	
	result = GetQuasiSequenceOrder(protein,maxlag,weight)
	
	Input: protein is a pure protein sequence
	
	maxlag is the maximum lag and the length of the protein should be larger
	
	than maxlag. default is 30.
	
	weight is a weight factor.  please see reference 1 for its choice. default is 0.1.
	
	Output: result is a dict form containing all quasi-sequence-order descriptors
	###############################################################################
	"""
	result=dict()
	result.update(GetQuasiSequenceOrder1SW(ProteinSequence,maxlag,weight,_Distance1))
	result.update(GetQuasiSequenceOrder2SW(ProteinSequence,maxlag,weight,_Distance1))
	result.update(GetQuasiSequenceOrder1Grant(ProteinSequence,maxlag,weight,_Distance2))
	result.update(GetQuasiSequenceOrder2Grant(ProteinSequence,maxlag,weight,_Distance2))
	return result
#############################################################################################
def GetQuasiSequenceOrderp(ProteinSequence,maxlag=30,weight=0.1,distancematrix=_Distance1):
	"""

	###############################################################################

	Computing quasi-sequence-order descriptors for a given protein.

	[1]:Kuo-Chen Chou. Prediction of Protein Subcellar Locations by 

	Incorporating Quasi-Sequence-Order Effect. Biochemical and Biophysical

	Research Communications 2000, 278, 477-483.

	Usage:

	result = GetQuasiSequenceOrderp(protein,maxlag,weight,distancematrix)

	Input: protein is a pure protein sequence

	maxlag is the maximum lag and the length of the protein should be larger

	than maxlag. default is 30.

	weight is a weight factor.  please see reference 1 for its choice. default is 0.1.
	
	distancematrix is a dict form containing 400 distance values

	Output: result is a dict form containing all quasi-sequence-order descriptors

	###############################################################################
	"""
	result=dict()
	result.update(GetQuasiSequenceOrder1(ProteinSequence,maxlag,weight,distancematrix))
	result.update(GetQuasiSequenceOrder2(ProteinSequence,maxlag,weight,distancematrix))
	return result
#############################################################################################
if __name__=="__main__":
    
    protein="ELRLRYCAPAGFALLKCNDADYDGFKTNCSNVSVVHCTNLMNTTVTTGLLLNGSYSENRT"
    
    #60
    SCN=GetSequenceOrderCouplingNumberTotal(protein,maxlag=30)
    print(len(SCN))
    
    SCNp = GetSequenceOrderCouplingNumberp(protein,maxlag=30)
    print(len(SCNp))
    
    #100
    QSO=GetQuasiSequenceOrder(protein,maxlag=30,weight=0.1) #has a bug: QSOSW same with QSOgrant
    print(len(QSO))
    
    QSOp=GetQuasiSequenceOrderp(protein,maxlag=30)
    print(len(QSOp))
