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
matplotlib.use('Agg')
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

modelN = sys.argv[1]
DIM = sys.argv[2]
op = sys.argv[3]
EPOCHS = 250
vaename = "/gwpool/users/glavizzari/Downloads/ML4Anomalies-main/myVAE/finalvae1_denselayers_latentdim"+str(DIM)+"_epoch"+str(EPOCHS)+"_batchsize32_log_eventFiltered"
#vaename = "finalvae"+str(modelN)+"_denselayers_latentdim"+str(DIM)+"_epoch"+str(EPOCHS)+"_batchsize32_log_eventFiltered"
#encname = "myenc"+str(modelN)+"_denselayers_latentdim"+str(DIM)+"_epoch"+str(EPOCHS)+"_batchsize32_log_eventFiltered"
#vaename = "../TRAINING/BESTvae_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered"

vae = tf.keras.models.load_model(vaename)
#enc = tf.keras.models.load_model(encname)

#print (vaename)
#print (encname)


cW = 0.3 #0.3

############################# taking in data and scaling #####################################

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll', 'w']#,'phij1', 'phij2', 'w']


dfAll = ROOT.RDataFrame("SSWW_SM","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_QUAD = ROOT.RDataFrame("SSWW_"+str(op)+"_QU","/gwpool/users/glavizzari/Downloads/ntuplesBSM/ntuple_SSWW_"+str(op)+"_QU.root")
dfBSM_QUAD = dfBSMAll_QUAD.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_LIN = ROOT.RDataFrame("SSWW_"+str(op)+"_LI","/gwpool/users/glavizzari/Downloads/ntuplesBSM/ntuple_SSWW_"+str(op)+"_LI.root")
dfBSM_LIN = dfBSMAll_LIN.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")

SM =pd.DataFrame.from_dict(df.AsNumpy(pd_variables))
BSM_quad=pd.DataFrame.from_dict(dfBSM_QUAD.AsNumpy(pd_variables))
BSM_lin=pd.DataFrame.from_dict(dfBSM_LIN.AsNumpy(pd_variables))

nEntries = 2000000000000000

SM = SM.head(nEntries)
BSM_quad = BSM_quad.head(nEntries)
BSM_lin = BSM_lin.head(nEntries)

#using logarithm of some variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    BSM_quad[vars] = BSM_quad[vars].apply(np.log10)
    BSM_lin[vars] = BSM_lin[vars].apply(np.log10)
    SM[vars] = SM[vars].apply(np.log10)

#print ("\nweights\n")

weightsSM = SM["w"].to_numpy()
weightsLIN = BSM_lin["w"].to_numpy()
weightsQUAD = BSM_quad["w"].to_numpy()

#print (weightsSM)
#print (weightsLIN)
#print (weightsQUAD)

SM.drop('w',axis='columns', inplace=True)
BSM_quad.drop('w',axis='columns', inplace=True)
BSM_lin.drop('w',axis='columns', inplace=True)

SM = SM.to_numpy()
LIN = BSM_lin.to_numpy()
QUAD = BSM_quad.to_numpy()

#print (SM)
#print (LIN)
#print (QUAD)

X_train, X_test, y_train, y_test = train_test_split(SM,SM,test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(weightsSM, weightsSM, test_size=0.2, random_state=1)

t = MinMaxScaler()
t.fit(X_train) 
X_train = t.transform(X_train)
X_test = t.transform(X_test)
SM = t.transform(SM)
LIN = t.transform(LIN)
QUAD = t.transform(QUAD)

#print (SM)
#print (LIN)
#print (QUAD)


################################## losses per batch ###################################

mylosses = LossPerBatch()
#vae.evaluate(X_test,X_test,batch_size=1,callbacks=[mylosses],verbose=0)

mylossesSM = LossPerBatch()
#vae.evaluate(SM,SM,batch_size=1,callbacks=[mylossesSM],verbose=0)

mylossesLIN = LossPerBatch()
vae.evaluate(LIN,LIN,batch_size=1,callbacks=[mylossesLIN],verbose=0)

mylossesQUAD = LossPerBatch()
vae.evaluate(QUAD,QUAD,batch_size=1,callbacks=[mylossesQUAD],verbose=0)


#myloss = mylosses.eval_loss
#mylossSM = mylossesSM.eval_loss
mylossLIN = mylossesLIN.eval_loss
mylossQUAD = mylossesQUAD.eval_loss

#myloss = np.asarray(myloss)
#mylossSM = np.asarray(mylossSM)
mylossLIN = np.asarray(mylossLIN)
mylossQUAD = np.asarray(mylossQUAD)

#np.savetxt("sm"+str(modelN)+"_dim"+str(DIM)+"_lossSM_split.csv", myloss,delimiter=',')
#np.savetxt("sm"+str(modelN)+"_dim"+str(DIM)+"_lossSM_total.csv", mylossSM,delimiter=',')
np.savetxt(str(op)+str(modelN)+"_dim"+str(DIM)+"_lossLIN_total.csv", mylossLIN,delimiter=',')
np.savetxt(str(op)+str(modelN)+"_dim"+str(DIM)+"_lossQUAD_total.csv", mylossQUAD,delimiter=',')

#np.savetxt("sm"+str(modelN)+"_dim"+str(DIM)+"_weightsSM_split.csv", wx_test,delimiter=',')
#np.savetxt("sm"+str(modelN)+"_dim"+str(DIM)+"_weightsSM_total.csv", weightsSM,delimiter=',')
np.savetxt(str(op)+str(modelN)+"_dim"+str(DIM)+"_weightsLIN_total.csv", weightsLIN,delimiter=',')
np.savetxt(str(op)+str(modelN)+"_dim"+str(DIM)+"_weightsQUAD_total.csv", weightsQUAD,delimiter=',')

'''
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function: model "+str(modelN)+", dim "+str(DIM))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(myloss,bins=250,range=[0., 0.4],weights=wx_test,histtype="step",color="orange",alpha=.3,linewidth=2,density =1,label ="SM test Loss")
ax.hist(mylossSM,bins=250,range=[0., 0.4],weights=weightsSM,histtype="step",color="crimson",alpha=.3,linewidth=2,density =1,label ="SM Loss")
ax.hist(mylossLIN,bins=250,range=[0., 0.4],weights=weightsLIN*cW,histtype="step",color="blue",alpha=.3,linewidth=2,density =1,label ="LIN Loss")
ax.hist(mylossQUAD,bins=250,range=[0., 0.4],weights=weightsQUAD*cW*cW,histtype="step",color="cornflowerblue",alpha=.3,linewidth=2,density =1,label ="QUAD Loss")
plt.legend(loc=1)
ax.patch.set_facecolor("w")
plt.savefig("./WandL/losses_m"+str(modelN)+"_dim"+str(DIM)+".png", bbox_inches='tight')
plt.close()
'''

print ("done")
