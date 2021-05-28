#plotting and thus comparing models with different latent dimensions (4, 5, 6)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ROOT
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
ROOT.ROOT.EnableImplicitMT()
# non ho preso la classe LossPerBatch... capire se serve.

grid = np.empty((3, 2))
i_ax = ["5", "6", "7"] #latent_dimension
j_ax = ["16","32"] #batch_size
DIM = [5, 6, 7] # i
BATCH = [16, 32] # j
EPOCHS = 200;


#grid structure:
#
#		5	6	7
#	16
#	32

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']

# cuts
dfAll = ROOT.RDataFrame("SSWW_SM","../ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")

npy = df.AsNumpy(pd_variables)
wSM = df.AsNumpy("w")
npd =pd.DataFrame.from_dict(npy)
wpdSM = pd.DataFrame.from_dict(wSM)

nEntries = 10000000
npd = npd.head(nEntries)
wpdSM = wpdSM.head(nEntries)

# log
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1', 'ptl2', 'ptll']:
	npd[vars] = npd[vars].apply(np.log10)


#splitting data
X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()

#scaling data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

#retrieving the LF (on test set) for every model
results = np.empty((3, 2))
print ("results.shape: ", results.shape)
vae1 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim5_epoch200_batchsize16_log_eventFiltered')
results[0,0] = vae1.evaluate(X_test, X_test, batch_size=16)
vae2 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim5_epoch200_batchsize32_log_eventFiltered')
results[0,1] = vae2.evaluate(X_test, X_test, batch_size=32)

vae4 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim6_epoch200_batchsize16_log_eventFiltered')
results[1,0] = vae4.evaluate(X_test, X_test, batch_size=16)
vae5 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim6_epoch200_batchsize32_log_eventFiltered')
results[1,1] = vae5.evaluate(X_test, X_test, batch_size=32)

vae7 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim7_epoch200_batchsize16_log_eventFiltered')
results[2,0] = vae7.evaluate(X_test, X_test, batch_size=16)
vae8 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered')
results[2,1] = vae8.evaluate(X_test, X_test, batch_size=32)


print ("\n\nComparing the models:\n")
for i in range (3):
	for j in range (2):
		print("\nLatent Dim: ", i_ax[i],"    Batch size: ", j_ax[j],"   Loss Fun (on testset): ", results[i,j], "  i=",i,"  j=",j)
		
minLoss = np.amin(results)
minLindex = np.unravel_index(np.argmin(results, axis=None), results.shape)
print ("\n\nThe minimum loss function obtained is: ", minLoss)
print ("\nWhich corresponds to: Latent dimension = ", i_ax[minLindex[0]], "     Batch size = ", j_ax[minLindex[1]],)
print ("\nindex : ", minLindex)




#############################################################################################
######################### LET'S ANALIZE THE BEST MODEL ######################################

dim = DIM[minLindex[0]]
batch = BATCH[minLindex[1]]
print ("\n\ndim: ", dim, "    batch: ", batch)

bestvae = "TRAINING/vae_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(dim, EPOCHS, batch)
bestenc = "TRAINING/enc_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(dim, EPOCHS, batch)
vae = tf.keras.models.load_model(bestvae)
enc = tf.keras.models.load_model(bestenc)

out = vae.predict(X_test)
latent = enc.predict(X_test)

nrows = 4
ncols = 4

#input vs reconstructed
fig1, axes1 = plt.subplots(nrows=4,ncols=4)
nvar1 = 0
fig1.suptitle("input vs reconstructed (SM only)")
for i in range(nrows):
    for j in range(ncols):
        if nvar1 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes1[i][j].hist(X_test[0:,nvar1],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="firebrick", alpha=0.8)
            axes1[i][j].hist(out[0:,nvar1],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes1[i][j].set_xlabel(pd_variables[nvar1])            
            nvar1=nvar1+1
patch1 = mpatches.Patch(color="firebrick", label="input")
patch2 = mpatches.Patch(color="midnightblue", label="reconstructed")
fig1.legend(handles=[patch1, patch2])
fig1.patch.set_facecolor("w")
fig1.subplots_adjust(hspace=0.845, right=0.824)

enc_variables = ["var1", "var2", "var3", "var4", "var5", "var6", "var7"]
nrowsenc = 2
ncolsenc = 4
fig3, axes3 = plt.subplots(nrows=2,ncols=4)
nvar3 = 0 
fig3.suptitle("Encoded Variables")
for i in range(nrowsenc):
    for j in range(ncolsenc):
        if nvar3 < len(enc_variables): #qui ho tolto i pesi
            axes3[i][j].hist(latent[0:,nvar3],bins=500, histtype="step", color="peru", alpha=0.8, label="Enc3 (dim7)")
            axes3[i][j].set_xlabel(enc_variables[nvar3])  
            nvar3= nvar3+1
fig3.subplots_adjust(hspace=0.845, right=0.824)


plt.show()



