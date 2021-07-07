import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import ROOT


cW = 0.3 #0.3
nEntries = 20

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll','w']#,'phij1', 'phij2']

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
#BSM_quad["Sample"] = 2 #adds a column ('sample') with value 2
#BSM_lin["Sample"] = 1
#SM["Sample"] = 0
All_BSM = pd.concat([BSM_quad, BSM_lin], keys=['Q','L'])


#using logarithm of some variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    All_BSM[vars] = All_BSM[vars].apply(np.log10)
    SM[vars] = SM[vars].apply(np.log10)

#Rescaling weights for the Wilson coefficient
All_BSM["w"].loc["L"] = All_BSM["w"].loc["L"].to_numpy()*cW
All_BSM["w"].loc["Q"] = All_BSM["w"].loc["Q"].to_numpy()*cW*cW

weights = All_BSM["w"].to_numpy()
weights_SM = SM["w"].to_numpy()

#concatenating the SM + BSM part
All = pd.concat([SM,All_BSM])

weights_all = All["w"].to_numpy()
SM.drop('w',axis='columns', inplace=True)
All_BSM.drop('w',axis='columns', inplace=True)
All.drop('w',axis='columns', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(SM,SM,test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(weights_SM, weights_SM, test_size=0.2, random_state=1)
#All_BSM = All_BSM.to_numpy()
_, All_BSM_test,_,_  = train_test_split(All_BSM,All_BSM,test_size=0.2, random_state=1)
w_train, w_test,_,_ = train_test_split(weights, weights,test_size=0.2, random_state=1)
All_test = np.concatenate((All_BSM_test,X_test))
weight_test = np.concatenate((w_test, wx_test))
#weight_test = np.abs(weight_test)

#t = StandardScaler()
t = MinMaxScaler()
t.fit(X_train) 
X_train = t.transform(X_train)
X_test=t.transform(X_test)
All_test = t.transform(All_test)

#DIM = sys.argv[1]
#print (DIM)

print ("variables\n", All_test)
print ("variables size: ", np.shape(All_test))

vae = tf.keras.models.load_model("./TRAINING/vae_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc = tf.keras.models.load_model("./TRAINING/enc_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
features = enc.predict(All_test)

vars_df=pd.DataFrame.from_dict(All_test)
feat_df=pd.DataFrame.from_dict(features)
pd_variables.remove('w')
vars_df.columns = pd_variables
feat_df.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']

vars_and_feat = vars_df.join(feat_df)
print (vars_and_feat)

#plotting correlation matrix
corrM = vars_and_feat.corr()
import seaborn as sn
ax = plt.axes()
sn.heatmap(corrM, annot=True, ax = ax)
ax.set_title('BSM + SM sample dim7')


featuresSM = enc.predict(X_test)
vars_df_SM=pd.DataFrame.from_dict(X_test)
feat_df_SM=pd.DataFrame.from_dict(featuresSM)
vars_df_SM.columns = pd_variables
feat_df_SM.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']

vars_and_feat_SM = vars_df_SM.join(feat_df_SM)
corrM_SM = vars_and_feat_SM.corr()
ax1 = plt.axes()
sn.heatmap(corrM_SM, annot=True, ax = ax1)
ax1.set_title('SM sample dim7')



plt.show()



