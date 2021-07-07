#python plotCombinedSamplesPROV.py model_name number_of_dimensions
# models on hercules: /gwpool/users/glavizzari/Downloads/ML4Anomalies-main/TRAINING
# ntuples on hercules: /gwpool/users/glavizzari/Downloads

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import ROOT
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

#from VAE_model import *

#ROOT.ROOT.EnableImplicitMT()

class LossPerBatch(tf.keras.callbacks.Callback):
    
    def __init__(self,**kwargs):
        self.eval_loss = []
    
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        self.eval_loss = []
        #print("Start predicting; got log keys: {}".format(keys))

    def on_test_batch_end(self, batch, logs=None):
        #print("For batch {}, loss is {:7.10f}.".format(batch, logs["loss"]))
        self.eval_loss.append(logs["loss"])
        

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )

cW = 0.3 #0.3
#cutLoss = 0.00004
nEntries = 100000000000000

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll',"w"]#,'phij1', 'phij2', 'w']

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
All_BSM = pd.concat([BSM_quad, BSM_lin], keys=['Q','L'])
'''
print (BSM_lin)
print ("Size of BSM_lin: ", BSM_lin.shape[0])
print (BSM_quad)
print ("Size of BSM_quad: ", BSM_quad.shape[0])
print (SM)
print ("Size of SM: ", SM.shape[0])
print (All_BSM)
print ("Size of All_BSM: ", All_BSM.shape[0])
'''

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

#plotting correlation matrix
SM_corrM = SM.corr()
All_corrM = All.corr()
corrMatrix = SM_corrM - All_corrM
import seaborn as sn
#sn.heatmap(corrMatrix, annot=True)
sn.heatmap(All_corrM, annot=True)

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

modelname = sys.argv[1]
print (modelname)

DIM = sys.argv[2]
print (DIM)

model = tf.keras.models.load_model(modelname)
mylosses = LossPerBatch()
mylosses_train = LossPerBatch()
model.evaluate(X_test,X_test,batch_size=1,callbacks=[mylosses],verbose=0)
model.evaluate(X_train,X_train,batch_size=1,callbacks=[mylosses_train],verbose=0)

mylosses_All = LossPerBatch()
model.evaluate(All_test,All_test,batch_size=1,callbacks=[mylosses_All],verbose=0)

myloss = mylosses.eval_loss
myloss_train = mylosses_train.eval_loss
myloss_All = mylosses_All.eval_loss

myloss =np.asarray(myloss)
myloss_All = np.asarray(myloss_All)
myloss_train =np.asarray(myloss_train)

print ("\n\n\nLOSSPERBATCH")
print (myloss_All)

np.savetxt("lossSM"+str(DIM)+"_noweigths_"+str(cW)+".csv", myloss,delimiter=',')
np.savetxt("lossBSM"+str(DIM)+"_noweigths_"+str(cW)+".csv", myloss_All,delimiter=',')
np.savetxt("weight_BSM"+str(DIM)+"_noweigths_"+str(cW)+".csv",weight_test,delimiter=',')
np.savetxt("weight_SM"+str(DIM)+"_noweigths_"+str(cW)+".csv",wx_test,delimiter=',')
##print "Eff All = ", 1.*(myloss_All>cutLoss).sum()/len(myloss_All)
##print "Eff SM = ",1.*(myloss>cutLoss).sum()/len(myloss)

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.set_ylim(ymax=10000)%
ax.hist(myloss_All,bins=250,range=(0.,0.05),weights=weight_test,histtype="step",color="red",alpha=.3,linewidth=2,density =1,label ="BSM Loss")
ax.hist(myloss,bins=250,range=(0.,0.05),weights=wx_test,histtype="step",color="blue",alpha=.3,linewidth=2,density =1, label="SM Test Loss")
ax.hist(myloss_train,bins=250,range=(0.,0.05),weights=wx_train,histtype="step",color="green",alpha=.3,linewidth=2,density =1, label="SM Train Loss")
#plt.hist(myloss_BSM2,bins=100,range=(0.,0.00015),histtype="step",color="green",alpha=1.)
plt.legend()
ax.patch.set_facecolor("w")
#plt.savefig('PCSloss.png', bbox_inches='tight')
plt.close()

#t = StandardScaler()
t = MinMaxScaler()
t.fit(SM)
SM = t.transform(SM)
All= t.transform(All)

#out = model.predict(All)
out = model.predict(All_test)
out_SM = model.predict(X_test)

diff = np.subtract(All_test, out)
#diff = np.abs(diff)
diffSM = np.subtract(X_test, out_SM)
#diffSM = np.abs(diffSM)

fig, axes = plt.subplots(nrows=4,ncols=4)
nvar = 0
nrows = 4
ncols = 4
fig.suptitle("difference in-out dim "+str(DIM))
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables)-1: 
            axes[i][j].hist(diff[0:,nvar],bins=500, range=[-1.,1.], weights =weight_test, histtype="step", color="midnightblue", alpha=0.6)
            axes[i][j].hist(diffSM[0:,nvar],bins=500, range=[-1.,1.], weights =wx_test, histtype="step", color="crimson", alpha=0.6)
            axes[i][j].set_xlabel(pd_variables[nvar])  
            axes[i][j].set_yscale('log')          
            nvar=nvar+1
fig.patch.set_facecolor("w")
fig.subplots_adjust(hspace=0.845, right=0.824)
patchB = mpatches.Patch(color="midnightblue", label="All(SM+BSM)")
patchR = mpatches.Patch(color="crimson", label="SM only")
fig.legend(handles=[patchB, patchR])
plt.savefig("PCSdiffhisto"+str(DIM)+".png", bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(nrows=4,ncols=4)
nvar = 0
nrows = 4
ncols = 4
fig.suptitle("difference vs variable dim "+str(DIM))
for i in range(nrows):
    for j in range(ncols):
        if nvar < len(pd_variables)-1: 
            axes[i][j].scatter(diff[0:,nvar], All_test[0:,nvar], s=4, color="midnightblue", alpha=0.3)
            axes[i][j].scatter(diffSM[0:,nvar], X_test[0:,nvar], s=4, color="crimson", alpha=0.3)
            axes[i][j].set_xlabel(pd_variables[nvar])   
            axes[i][j].set_xlim(-1., 1.8) # -1., 1. per min max scaler
            axes[i][j].set_ylim(-0.3, 1.3)    
            nvar=nvar+1
fig.patch.set_facecolor("w")
fig.subplots_adjust(hspace=0.845, right=0.824)
fig.legend(handles=[patchB, patchR])
plt.savefig("PCSdiffscatter"+str(DIM)+".png", bbox_inches='tight')
plt.close(fig)


print ("done")
#plt.show()
