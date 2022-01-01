from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import sys
import numpy
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


#taking the model
#from VAE_model_extended_moreDKL import *
from VAE_testReturnLoss import *
from matplotlib import pyplot as plt

import ROOT
#ROOT.ROOT.EnableImplicitMT()




#
# variable from the nutple
#
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
kinematicFilter = "ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200"
dfSM = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
dfSM = dfSM.Filter(kinematicFilter)
dfBSM = ROOT.RDataFrame("SSWW_cW_QU","../ntuple_SSWW_cW_QU.root")
dfBSM = dfBSM.Filter(kinematicFilter)

np_SM = dfSM.AsNumpy(pd_variables)
wSM = dfSM.AsNumpy("w")
npd =pd.DataFrame.from_dict(np_SM)
wpdSM = pd.DataFrame.from_dict(wSM)

np_BSM = dfBSM.AsNumpy(pd_variables)
wBSM = dfBSM.AsNumpy("w")
npd_BSM =pd.DataFrame.from_dict(np_BSM)
wpdBSM = pd.DataFrame.from_dict(wBSM)

nEntries = 300000
npd = npd.head(nEntries)
npd_BSM = npd_BSM.head(nEntries)
wpdSM = wpdSM.head(nEntries)
wpdBSM = wpdBSM.head(nEntries)
#to be done for all the pt and mass and met variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    npd[vars] = npd[vars].apply(numpy.log10)
    npd_BSM[vars] = npd_BSM[vars].apply(numpy.log10)

Y_true = np.full(nEntries,0)
Y_true_BSM = np.full(nEntries,1)

X_train, X_test, y_train, y_test = train_test_split(npd, Y_true, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)

BSM_train, BSM_test, y_BSM_train, y_BSM_test = train_test_split(npd_BSM, Y_true_BSM, test_size=0.2, random_state=1)
wBSM_train, wBSM_test, _ , _ = train_test_split(wpdBSM, wpdBSM, test_size=0.2, random_state=1)
#print wx_train,X_train
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
wBSM = wBSM_train["w"].to_numpy()
wBSMtest = wBSM_test["w"].to_numpy()
# scale data
t = MinMaxScaler()
#t = StandardScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
BSM_train = t.transform(BSM_train)
BSM_test = t.transform(BSM_test)

n_inputs = npd.shape[1]
original_dim = n_inputs

intermediate_dim = 20 #50 by default
input_dim = 10 #was 20 in default
half_input = 7 #was 20 in the newTest
latent_dim = 3 #was 3 for optimal performance
epochs = 200 #100
batchsize=64 #32
nameExtenstion = str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs)+"_"+str(batchsize)
vae = VariationalAutoEncoder(original_dim,intermediate_dim,input_dim,half_input,latent_dim)  
#vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())
#vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),run_eagerly=True, loss="binary_crossentropy",metrics = [tf.keras.metrics.BinaryAccuracy()])
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss="binary_crossentropy",metrics = [tf.keras.metrics.BinaryAccuracy()])

#concatenating SM and BSM
train_sample = np.concatenate((X_train,BSM_train))
test_sample = np.concatenate((X_test,BSM_test))

label_train = np.concatenate((y_train,y_BSM_train))
label_test = np.concatenate((y_test,y_BSM_test))

hist = vae.fit(train_sample, label_train,validation_data=(test_sample,label_test), epochs=epochs, batch_size = batchsize) 
#encoder = LatentSpace(intermediate_dim,input_dim,half_input,latent_dim)
#z = encoder.predict(X_train)
#tf.keras.models.save_model(encoder,'latent_newTest_newModelDimenstions_MinMaxScaler_'+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs))
tf.keras.models.save_model(vae,'vae_test_newModelUsingLatentSpace_'+nameExtenstion)
#numpy.savetxt("lossVAE_test_newModelDimenstions_MinMaxScaler_"+nameExtenstion+".csv",hist.history["loss"],delimiter=",")
#vae=tf.keras.models.load_model('vae_test_newModelUsingLatentSpace_'+nameExtenstion)


output_SM = vae.predict(X_test)
output_BSM = vae.predict(BSM_test)

print output_SM
print output_BSM
bins=100
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(output_SM,bins=bins, density=1,range=[0.,1.],histtype="step",color="red",alpha=0.6,linewidth=2,label="SM Output")                        
ax.hist(output_BSM,bins=bins, density=1,range=[0.,1.],histtype="step",color="blue",alpha=0.6,linewidth=2,label="BSM Output")                        
plt.rc('legend',fontsize='xx-small')    
plt.show()
