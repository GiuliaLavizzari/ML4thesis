import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import ROOT


cW = 0.5 #0.3
nEntries = 500000

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll',"w"]#,'phij1', 'phij2']

dfAll = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_QUAD = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")
dfBSM_QUAD = dfBSMAll_QUAD.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_LIN = ROOT.RDataFrame("SSWW_cW_LI","../ntuple_SSWW_cW_LI.root")
dfBSM_LIN = dfBSMAll_LIN.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")

SM =pd.DataFrame.from_dict(df.AsNumpy(pd_variables))
BSM_quad=pd.DataFrame.from_dict(dfBSMAll_QUAD.AsNumpy(pd_variables))
BSM_lin=pd.DataFrame.from_dict(dfBSMAll_LIN.AsNumpy(pd_variables))
SM = SM.head(nEntries)
BSM_lin = BSM_lin.head(nEntries)
BSM_quad = BSM_quad.head(nEntries)
BSM_quad["Sample"] = 2 #adds a column ('sample') with value 2
BSM_lin["Sample"] = 1
SM["Sample"] = 0
All_BSM = pd.concat([BSM_quad, BSM_lin], keys=['Q','L'])

All = pd.concat([SM,BSM_quad, BSM_lin], keys=["S","Q","L"])
All_corrM = All.corr()
import seaborn as sn
#sn.heatmap(corrMatrix, annot=True)
sn.heatmap(All_corrM, annot=True)
plt.show()




