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
        print ("A was called for")
    
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        self.eval_loss = []
        print ("B was called for")
        #print("Start predicting; got log keys: {}".format(keys))

    def on_test_batch_end(self, batch, logs=None):
        #print("For batch {}, loss is {:7.10f}.".format(batch, logs["loss"]))
        self.eval_loss.append(logs["loss"])######################
        #print ("C was called for")
        

    def on_epoch_end(self, epoch, logs=None):
        print ("D was called for")
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )



modelN = sys.argv[1]
DIM = sys.argv[2]
EPOCHS = 200
vaename = "myvae"+str(modelN)+"_denselayers_latentdim"+str(DIM)+"_epoch"+str(EPOCHS)+"_batchsize32_log_eventFiltered"
encname = "myenc"+str(modelN)+"_denselayers_latentdim"+str(DIM)+"_epoch"+str(EPOCHS)+"_batchsize32_log_eventFiltered"

vae = tf.keras.models.load_model(vaename)
enc = tf.keras.models.load_model(encname)

print (vaename)
print (encname)


cW = 0.3 #0.3
nEntries = 50000#0000000000000



############################# taking in data and scaling #####################################

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll',"w"]#,'phij1', 'phij2', 'w']

dfAll = ROOT.RDataFrame("SSWW_SM","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_QUAD = ROOT.RDataFrame("SSWW_cW_QU","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_cW_QU.root")
dfBSM_QUAD = dfBSMAll_QUAD.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_LIN = ROOT.RDataFrame("SSWW_cW_LI","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_cW_LI.root")
dfBSM_LIN = dfBSMAll_LIN.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")

SM =pd.DataFrame.from_dict(df.AsNumpy(pd_variables))
BSM_quad=pd.DataFrame.from_dict(dfBSMAll_QUAD.AsNumpy(pd_variables))
BSM_lin=pd.DataFrame.from_dict(dfBSMAll_LIN.AsNumpy(pd_variables))
SM = SM.head(nEntries)
BSM_lin = BSM_lin.head(nEntries)
BSM_quad = BSM_quad.head(nEntries)
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

#t = StandardScaler()
t = MinMaxScaler()
t.fit(X_train) 
X_train = t.transform(X_train)
X_test=t.transform(X_test)
All_test = t.transform(All_test)

print ("size of X_test: ", np.shape(X_test))

################################## losses per batch ###################################

print ("\n\n\n############ Total loss ##############\n")

mylosses = LossPerBatch()
vae.evaluate(X_test,X_test,batch_size=1,callbacks=[mylosses],verbose=0)

myloss = mylosses.eval_loss
myloss =np.asarray(myloss)
#myloss =np.multiply(myloss, wx_test)

print ("\n\nTotal loss:\n", myloss)


print ("\n\n\n############ MSE ##############\n")

mse = tf.keras.losses.MeanSquaredError()
myloss_mse = []
out = vae.predict(X_test)

for i in range(len(X_test)):
    loss = mse(out[i],X_test[i]).numpy()
    myloss_mse.append(loss)
   
#print ("\n\nLOSS MSE:\n",myloss_mse)   



print ("\n\n\n############ KL ##############\n")

latent = enc.predict(X_test)
#print (latent)
#print ("shape latent", np.shape(latent))

epsilon = (tf.keras.backend.random_normal(shape=(1,7))).numpy()
#print (epsilon)

kl = tf.keras.losses.KLDivergence()
myloss_kl = []

for i in range(len(X_test)):
    loss = (kl(latent[i],epsilon).numpy())
    myloss_kl.append(loss)
   
#print ("\n\nLOSS KL:\n",myloss_kl)



'''
print ("############################")
epsilon1 = tf.keras.backend.random_normal(shape=(len(latent),1))
print (epsilon1)
print (latent[:,2])
print (np.shape(latent[:,2]))
klperfeat = []

for i in range(7):
	loss = (kl(latent[:,i], epsilon1).numpy())/100000000.
	klperfeat.append(loss)
	
print ("kl per feat", klperfeat)
'''
print ("\n\n\n############################\n")
print ("KL PER FEATURE\n\n")
trans = np.transpose(latent)
epsilon1 = tf.keras.backend.random_normal(shape=(len(latent),1))
epsilon1 = np.transpose(epsilon1)
#print (trans)
#print (np.shape(trans))
#print (epsilon1)

klperfeat = []

for i in range(len(trans)):
	loss = (kl(trans[i], epsilon1).numpy())
	klperfeat.append(loss)

print ("kl per feat", klperfeat)

meankl = np.mean(klperfeat)
print ("\nmean kl per model: ", meankl)




print ("\n\n\n############ SUMMARY ##############\n")

diff = np.subtract(myloss, myloss_mse)

pdlossesw = pd.DataFrame()

pdlossesw["all"] = np.multiply(myloss, wx_test)
pdlossesw["mse"] = np.multiply(myloss_mse, wx_test)
pdlossesw["kld"] = np.multiply(myloss_kl, wx_test)
pdlossesw["dif"] = np.multiply(diff, wx_test)
allM = np.mean(np.multiply(myloss, wx_test))
mseM = np.mean(np.multiply(myloss_mse, wx_test))
kldM = np.mean(np.multiply(myloss_kl, wx_test))
difM = np.mean(np.multiply(diff, wx_test))

print ("\n\nALL MY LOSSES weighted\n", pdlossesw)

print ("mean all: ", allM)
print ("mean mse: ", mseM)
print ("mean kld: ", kldM)
print ("mean dif: ", difM)

pdlosses = pd.DataFrame()

pdlosses["all"] = myloss
pdlosses["mse"] = myloss_mse
pdlosses["kld"] = myloss_kl
pdlosses["dif"] = diff

print ("\n\nALL MY LOSSES\n", pdlosses)



