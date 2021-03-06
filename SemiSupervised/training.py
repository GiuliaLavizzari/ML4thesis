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
from VAE_testDK_Reco_Loss import *
KIND = "A"
# A = latent space
# B = loss 1D (Reco)
# C = loss 2D (Reco+KLD)

from matplotlib import pyplot as plt

import ROOT
#ROOT.ROOT.EnableImplicitMT()


path_to_ntuple = "/home/giulialavizzari/Downloads/ntuples"

#
# variable from the nutple
#
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
kinematicFilter = "ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200"
dfSM = ROOT.RDataFrame("SSWW_SM",path_to_ntuple+"/ntuple_SSWW_SM.root")
dfSM = dfSM.Filter(kinematicFilter)
dfBSM = ROOT.RDataFrame("SSWW_cW_QU", path_to_ntuple+"/ntuple_SSWW_cW_QU.root")
dfBSM = dfBSM.Filter(kinematicFilter)

np_SM = dfSM.AsNumpy(pd_variables)
wSM = dfSM.AsNumpy("w")
npd =pd.DataFrame.from_dict(np_SM)
wpdSM = pd.DataFrame.from_dict(wSM)

np_BSM = dfBSM.AsNumpy(pd_variables)
wBSM = dfBSM.AsNumpy("w")
npd_BSM =pd.DataFrame.from_dict(np_BSM)
wpdBSM = pd.DataFrame.from_dict(wBSM)

nEntries = 3000#00
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
#concatenating SM and BSM
samples = np.concatenate((npd,npd_BSM))
labels = np.concatenate((Y_true,Y_true_BSM))

X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=1)
SM_train,SM_test,_,_ = train_test_split(npd, npd, test_size=0.2, random_state=1)
BSM_train,BSM_test,_,_ = train_test_split(npd_BSM, npd_BSM, test_size=0.2, random_state=1)
#wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)

#BSM_train, BSM_test, y_BSM_train, y_BSM_test = train_test_split(npd_BSM, Y_true_BSM, test_size=0.2, random_state=1)
#wBSM_train, wBSM_test, _ , _ = train_test_split(wpdBSM, wpdBSM, test_size=0.2, random_state=1)
#print wx_train,X_train
#wx = wx_train["w"].to_numpy()
#wxtest = wx_test["w"].to_numpy()
#wBSM = wBSM_train["w"].to_numpy()
#wBSMtest = wBSM_test["w"].to_numpy()
# scale data

t = MinMaxScaler()
#t = StandardScaler()
t.fit(SM_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

n_inputs = npd.shape[1]
original_dim = n_inputs

intermediate_dim = 20 #50 by default
input_dim = 10 #was 20 in default
half_input = 7 #was 20 in the newTest
latent_dim = 3 #was 3 for optimal performance
epochs = 2#80 #80
batchsize=64 #32
nameExtenstion = str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs)+"_"+str(batchsize)
vae = VariationalAutoEncoder(original_dim,intermediate_dim,input_dim,half_input,latent_dim)  
#vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),  loss=tf.keras.losses.MeanSquaredError())
#vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),run_eagerly=True, loss="binary_crossentropy",metrics = [tf.keras.metrics.BinaryAccuracy()])
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss="binary_crossentropy",metrics = [tf.keras.metrics.BinaryAccuracy()])

hist = vae.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=epochs, batch_size = batchsize) 
#print "new model: ", vae.summary()
encoderDecoder =  EncoderDecoder(original_dim,intermediate_dim,input_dim,half_input,latent_dim)
reco = encoderDecoder.predict(X_test)
#encoder = LatentSpace(intermediate_dim,input_dim,half_input,latent_dim)
#z = encoder.predict(X_train)
tf.keras.models.save_model(encoderDecoder,'encdec'+str(KIND)+"_"+nameExtenstion)
tf.keras.models.save_model(vae,'vae'+str(KIND)+"_"+nameExtenstion)

myloss = np.stack((hist.history["loss"], hist.history["val_loss"], hist.history["binary_accuracy"], hist.history["val_binary_accuracy"]), axis=1)

numpy.savetxt("loss"+str(KIND)+"_"+nameExtenstion+".csv", myloss)
#vae=tf.keras.models.load_model('vae_test_newModelUsingLatentSpace_'+nameExtenstion)


output_SM = vae.predict(X_test)
output_BSM = vae.predict(BSM_test)

#print output_SM
#print output_BSM
bins=100
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(output_SM,bins=bins, density=1,range=[0.,1.],histtype="step",color="red",alpha=0.6,linewidth=2,label="SM Output")                        
ax.hist(output_BSM,bins=bins, density=1,range=[0.,1.],histtype="step",color="blue",alpha=0.6,linewidth=2,label="BSM Output")                        
plt.rc('legend',fontsize='small')    
plt.show()


x_bins = range(len(hist.history["loss"]))
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(x_bins,hist.history["loss"],color="blue",label="training losss")                        
plt.rc('legend',fontsize='small')    
plt.show()
