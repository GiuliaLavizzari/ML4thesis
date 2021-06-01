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

print ("\n\nvae1 = latentdim5_epoch200_batchsize16   [0,0]")
vae1 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim5_epoch200_batchsize16_log_eventFiltered')
results[0,0] = vae1.evaluate(X_test, X_test, batch_size=16)

print ("\n\nvae2 = latentdim5_epoch200_batchsize32   [0,1]")
vae2 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim5_epoch200_batchsize32_log_eventFiltered')
results[0,1] = vae2.evaluate(X_test, X_test, batch_size=32)

print ("\n\nvae4 = latentdim6_epoch200_batchsize16   [1,0]")
vae4 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim6_epoch200_batchsize16_log_eventFiltered')
results[1,0] = vae4.evaluate(X_test, X_test, batch_size=16)

print ("\n\nvae5 = latentdim6_epoch200_batchsize32   [1,1]")
vae5 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim6_epoch200_batchsize32_log_eventFiltered')
results[1,1] = vae5.evaluate(X_test, X_test, batch_size=32)

print ("\n\nvae7 = latentdim7_epoch200_batchsize16   [2,0]")
vae7 = tf.keras.models.load_model('TRAINING/vae_denselayers_latentdim7_epoch200_batchsize16_log_eventFiltered')
results[2,0] = vae7.evaluate(X_test, X_test, batch_size=16)

print ("\n\nvae8 = latentdim7_epoch200_batchsize32   [2,1]")
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

print ("\n\n\n")

##################################################################################
################## REMOVING VARS: LET'S SEE WHAT HAPPENS #########################
# working with dim = 7, epochs = 200 and batch = 32 

############################ etaj1 removed ###########################
'''
vae_etaj1 = tf.keras.models.load_model("TRAINING/vae_etaj1_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_etaj1 = tf.keras.models.load_model("TRAINING/enc_etaj1_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_etaj1 = np.delete(X_test, 2, 1)

out_etaj1 = vae_etaj1.predict(X_test_etaj1)
results_etaj1 = vae_etaj1.evaluate(X_test_etaj1, X_test_etaj1, batch_size=32)
print ("Loss function w/o etaj1: ", results_etaj1)

originalout_etaj1 = np.delete(out, 2, 1)
del pd_variables[2]

fig2, axes2 = plt.subplots(nrows=4,ncols=4)
nvar2 = 0
fig2.suptitle("in vs out (SM only) w/o etaj1")
for i in range(nrows):
    for j in range(ncols):
        if nvar2 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes2[i][j].hist(X_test_etaj1[0:,nvar2],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes2[i][j].hist(originalout_etaj1[0:,nvar2],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes2[i][j].hist(out_etaj1[0:,nvar2],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes2[i][j].set_xlabel(pd_variables[nvar2])            
            nvar2=nvar2+1
'''
patch_1 = mpatches.Patch(color="crimson", label="input")
patch_2 = mpatches.Patch(color="midnightblue", label="out old")
patch_3 = mpatches.Patch(color="deepskyblue", label="out new")
'''
fig2.legend(handles=[patch_1, patch_2, patch_3])
fig2.patch.set_facecolor("w")
fig2.subplots_adjust(hspace=0.845, right=0.824)

pd_variables.insert(2, 'etaj1')
print ("\n\n")


############################ etaj2 removed ###########################

vae_etaj2 = tf.keras.models.load_model("TRAINING/vae_etaj2_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_etaj2 = tf.keras.models.load_model("TRAINING/enc_etaj2_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_etaj2 = np.delete(X_test, 3, 1)

out_etaj2 = vae_etaj2.predict(X_test_etaj2)
results_etaj2 = vae_etaj2.evaluate(X_test_etaj2, X_test_etaj2, batch_size=32)
print ("Loss function w/o etaj2: ", results_etaj2)

originalout_etaj2 = np.delete(out, 3, 1)
del pd_variables[3]

fig3, axes3 = plt.subplots(nrows=4,ncols=4)
nvar3 = 0
fig3.suptitle("in vs out (SM only) w/o etaj2")
for i in range(nrows):
    for j in range(ncols):
        if nvar3 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes3[i][j].hist(X_test_etaj2[0:,nvar3],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes3[i][j].hist(originalout_etaj2[0:,nvar3],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes3[i][j].hist(out_etaj2[0:,nvar3],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes3[i][j].set_xlabel(pd_variables[nvar3])            
            nvar3=nvar3+1
fig3.legend(handles=[patch_1, patch_2, patch_3])
fig3.patch.set_facecolor("w")
fig3.subplots_adjust(hspace=0.845, right=0.824)

pd_variables.insert(3, 'etaj2')
print ("\n\n")

############################ met removed ###########################

vae_met = tf.keras.models.load_model("TRAINING/vae_met_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_met = tf.keras.models.load_model("TRAINING/enc_met_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_met = np.delete(X_test, 6, 1)

out_met = vae_met.predict(X_test_met)
results_met = vae_met.evaluate(X_test_met, X_test_met, batch_size=32)
print ("Loss function w/o met: ", results_met)

originalout_met = np.delete(out, 6, 1)
del pd_variables[6]

fig4, axes4 = plt.subplots(nrows=4,ncols=4)
nvar4 = 0
fig4.suptitle("in vs out (SM only) w/o met")
for i in range(nrows):
    for j in range(ncols):
        if nvar4 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes4[i][j].hist(X_test_met[0:,nvar4],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes4[i][j].hist(originalout_met[0:,nvar4],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes4[i][j].hist(out_met[0:,nvar4],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes4[i][j].set_xlabel(pd_variables[nvar4])            
            nvar4=nvar4+1
fig4.legend(handles=[patch_1, patch_2, patch_3])
fig4.patch.set_facecolor("w")
fig4.subplots_adjust(hspace=0.845, right=0.824)

pd_variables.insert(6, 'met')
print ("\n\n")


############################ ptll removed ###########################

vae_ptll = tf.keras.models.load_model("TRAINING/vae_ptll_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_ptll = tf.keras.models.load_model("TRAINING/enc_ptll_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_ptll = np.delete(X_test, 13, 1)

out_ptll = vae_ptll.predict(X_test_ptll)
results_ptll = vae_ptll.evaluate(X_test_ptll, X_test_ptll, batch_size=32)
print ("Loss function w/o ptll: ", results_ptll)

originalout_ptll = np.delete(out, 13, 1)
del pd_variables[13]

fig5, axes5 = plt.subplots(nrows=4,ncols=4)
nvar5 = 0
fig5.suptitle("in vs out (SM only) w/o ptll")
for i in range(nrows):
    for j in range(ncols):
        if nvar5 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes5[i][j].hist(X_test_ptll[0:,nvar5],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes5[i][j].hist(originalout_ptll[0:,nvar5],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes5[i][j].hist(out_ptll[0:,nvar5],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes5[i][j].set_xlabel(pd_variables[nvar5])            
            nvar5=nvar5+1
fig5.legend(handles=[patch_1, patch_2, patch_3])
fig5.patch.set_facecolor("w")
fig5.subplots_adjust(hspace=0.845, right=0.824)

pd_variables.insert(13, 'ptll')
print ("\n\n")

'''

##############################################################################
############################ etaj1 & etaj2 removed ###########################

vae_A = tf.keras.models.load_model("TRAINING/vae_A_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_A = tf.keras.models.load_model("TRAINING/enc_A_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_A = np.delete(X_test, 2, 1)
X_test_A = np.delete(X_test_A, 2, 1)

out_A = vae_A.predict(X_test_A)
results_A = vae_A.evaluate(X_test_A, X_test_A, batch_size=32)
print ("Loss function w/o A: ", results_A)

originalout_A = np.delete(out, 2, 1)
originalout_A = np.delete(originalout_A, 2, 1)
del pd_variables[2]
del pd_variables[2]

#print (" var tolte ", pd_variables)

fig6, axes6 = plt.subplots(nrows=4,ncols=4)
nvar6 = 0
fig6.suptitle("in vs out (SM only) w/o etaj1 & etaj2")
for i in range(nrows):
    for j in range(ncols):
        if nvar6 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes6[i][j].hist(X_test_A[0:,nvar6],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes6[i][j].hist(originalout_A[0:,nvar6],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes6[i][j].hist(out_A[0:,nvar6],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes6[i][j].set_xlabel(pd_variables[nvar6])            
            nvar6=nvar6+1
fig6.legend(handles=[patch_1, patch_2, patch_3])
fig6.patch.set_facecolor("w")
fig6.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")


############################ etaj1 & etaj2 & met removed ###########################

vae_B = tf.keras.models.load_model("TRAINING/vae_B_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_B = tf.keras.models.load_model("TRAINING/enc_B_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_B = np.delete(X_test_A, 4, 1)

out_B = vae_B.predict(X_test_B)
results_B = vae_B.evaluate(X_test_B, X_test_B, batch_size=32)
print ("Loss function w/o B: ", results_B)

originalout_B = np.delete(originalout_A, 4, 1)
del pd_variables[4]


#print (" var tolte ", pd_variables)

fig7, axes7 = plt.subplots(nrows=4,ncols=4)
nvar7 = 0
fig7.suptitle("in vs out (SM only) w/o etaj1 & etaj2 & met")
for i in range(nrows):
    for j in range(ncols):
        if nvar7 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes7[i][j].hist(X_test_B[0:,nvar7],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes7[i][j].hist(originalout_B[0:,nvar7],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes7[i][j].hist(out_B[0:,nvar7],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes7[i][j].set_xlabel(pd_variables[nvar7])            
            nvar7=nvar7+1
fig7.legend(handles=[patch_1, patch_2, patch_3])
fig7.patch.set_facecolor("w")
fig7.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")


############################ etaj1 & etaj2 & met & ptll removed ###########################

vae_C = tf.keras.models.load_model("TRAINING/vae_C_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_C = tf.keras.models.load_model("TRAINING/enc_C_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_C = np.delete(X_test_B, 10, 1)

out_C = vae_C.predict(X_test_C)
results_C = vae_C.evaluate(X_test_C, X_test_C, batch_size=32)
print ("Loss function w/o C: ", results_C)

originalout_C = np.delete(originalout_B, 10, 1)
del pd_variables[10]


#print (" var tolte ", pd_variables)

fig8, axes8 = plt.subplots(nrows=4,ncols=4)
nvar8 = 0
fig8.suptitle("in vs out (SM only) w/o etaj1 & etaj2 & met & ptll")
for i in range(nrows):
    for j in range(ncols):
        if nvar8 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes8[i][j].hist(X_test_C[0:,nvar8],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes8[i][j].hist(originalout_C[0:,nvar8],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes8[i][j].hist(out_C[0:,nvar8],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes8[i][j].set_xlabel(pd_variables[nvar8])            
            nvar8=nvar8+1
fig8.legend(handles=[patch_1, patch_2, patch_3])
fig8.patch.set_facecolor("w")
fig8.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")

pd_variables.insert(10, 'ptll')
pd_variables.insert(4, 'met')
pd_variables.insert(2, 'etaj2')
pd_variables.insert(2, 'etaj1')

'''
##############################################################################
############################ met & ptll removed ###########################

vae_D = tf.keras.models.load_model("TRAINING/vae_D_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_D = tf.keras.models.load_model("TRAINING/enc_D_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_D = np.delete(X_test, 6, 1)
X_test_D = np.delete(X_test_D, 12, 1)

out_D = vae_D.predict(X_test_D)
results_D = vae_D.evaluate(X_test_D, X_test_D, batch_size=32)
print ("Loss function w/o D: ", results_D)

originalout_D = np.delete(out, 6, 1)
originalout_D = np.delete(originalout_D, 12, 1)
del pd_variables[6]
del pd_variables[12]

#print (" var tolte ", pd_variables) 

fig7, axes7 = plt.subplots(nrows=4,ncols=4)
nvar7 = 0
fig7.suptitle("in vs out (SM only) w/o met & ptll")
for i in range(nrows):
    for j in range(ncols):
        if nvar7 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes7[i][j].hist(X_test_D[0:,nvar7],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes7[i][j].hist(originalout_D[0:,nvar7],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes7[i][j].hist(out_D[0:,nvar7],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes7[i][j].set_xlabel(pd_variables[nvar7])            
            nvar7=nvar7+1
fig7.legend(handles=[patch_1, patch_2, patch_3])
fig7.patch.set_facecolor("w")
fig7.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")

pd_variables.insert(12, 'ptll')
pd_variables.insert(6, 'met')


############################ etaj2 & met removed ###########################

vae_E = tf.keras.models.load_model("TRAINING/vae_E_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_E = tf.keras.models.load_model("TRAINING/enc_E_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_E = np.delete(X_test, 3, 1)
X_test_E = np.delete(X_test_E, 5, 1)

out_E = vae_E.predict(X_test_E)
results_E = vae_E.evaluate(X_test_E, X_test_E, batch_size=32)
print ("Loss function w/o E: ", results_E)

originalout_E = np.delete(out, 3, 1)
originalout_E = np.delete(originalout_E, 5, 1)
del pd_variables[3]
del pd_variables[5]

#print (" var tolte ", pd_variables)

fig8, axes8 = plt.subplots(nrows=4,ncols=4)
nvar8 = 0
fig8.suptitle("in vs out (SM only) w/o etaj2 & met")
for i in range(nrows):
    for j in range(ncols):
        if nvar8 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes8[i][j].hist(X_test_E[0:,nvar8],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes8[i][j].hist(originalout_E[0:,nvar8],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes8[i][j].hist(out_E[0:,nvar8],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes8[i][j].set_xlabel(pd_variables[nvar8])            
            nvar8=nvar8+1
fig8.legend(handles=[patch_1, patch_2, patch_3])
fig8.patch.set_facecolor("w")
fig8.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")

pd_variables.insert(5, 'met')
pd_variables.insert(3, 'etaj2')



############################ etaj2 & met removed ###########################

vae_F = tf.keras.models.load_model("TRAINING/vae_F_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")
enc_F = tf.keras.models.load_model("TRAINING/enc_F_denselayers_latentdim7_epoch200_batchsize32_log_eventFiltered")

X_test_F = np.delete(X_test, 2, 1)
X_test_F = np.delete(X_test_F, 5, 1)

out_F = vae_F.predict(X_test_F)
results_F = vae_F.evaluate(X_test_F, X_test_F, batch_size=32)
print ("Loss function w/o F: ", results_F)

originalout_F = np.delete(out, 2, 1)
originalout_F = np.delete(originalout_F, 5, 1)
del pd_variables[2]
del pd_variables[5]

#print (" var tolte ", pd_variables)

fig9, axes9 = plt.subplots(nrows=4,ncols=4)
nvar9 = 0
fig9.suptitle("in vs out (SM only) w/o etaj1 & met")
for i in range(nrows):
    for j in range(ncols):
        if nvar9 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes9[i][j].hist(X_test_F[0:,nvar9],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes9[i][j].hist(originalout_F[0:,nvar9],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes9[i][j].hist(out_F[0:,nvar9],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes9[i][j].set_xlabel(pd_variables[nvar9])            
            nvar9=nvar9+1
fig9.legend(handles=[patch_1, patch_2, patch_3])
fig9.patch.set_facecolor("w")
fig9.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")

pd_variables.insert(5, 'met')
pd_variables.insert(2, 'etaj1')

'''
######################################## dim 6 #####################################

############################ etaj1 & etaj2 removed ###########################

vae_A1 = tf.keras.models.load_model("TRAINING/vae_A_denselayers_latentdim6_epoch200_batchsize32_log_eventFiltered")
enc_A1 = tf.keras.models.load_model("TRAINING/enc_A_denselayers_latentdim6_epoch200_batchsize32_log_eventFiltered")

#X_test_A = np.delete(X_test, 2, 1)
#X_test_A = np.delete(X_test_A, 2, 1)

out_A1 = vae_A1.predict(X_test_A)
results_A1 = vae_A1.evaluate(X_test_A, X_test_A, batch_size=32)
print ("Loss function w/o A (dim6): ", results_A1)

#originalout_A = np.delete(out, 2, 1)
#originalout_A = np.delete(originalout_A, 2, 1)
del pd_variables[2]
del pd_variables[2]

#print (" var tolte ", pd_variables)

fig12, axes12 = plt.subplots(nrows=4,ncols=4)
nvar12 = 0
fig12.suptitle("in vs out (SM only) w/o etaj1 & etaj2 (dim 6)")
for i in range(nrows):
    for j in range(ncols):
        if nvar12 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes12[i][j].hist(X_test_A[0:,nvar12],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes12[i][j].hist(originalout_A[0:,nvar12],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes12[i][j].hist(out_A1[0:,nvar12],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes12[i][j].set_xlabel(pd_variables[nvar12])            
            nvar12=nvar12+1
fig12.legend(handles=[patch_1, patch_2, patch_3])
fig12.patch.set_facecolor("w")
fig12.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")




############################ etaj1 & etaj2 & met removed ###########################

vae_B1 = tf.keras.models.load_model("TRAINING/vae_B_denselayers_latentdim6_epoch200_batchsize32_log_eventFiltered")
enc_B1 = tf.keras.models.load_model("TRAINING/enc_B_denselayers_latentdim6_epoch200_batchsize32_log_eventFiltered")

#X_test_B = np.delete(X_test_A, 4, 1)

out_B1 = vae_B1.predict(X_test_B)
results_B1 = vae_B1.evaluate(X_test_B, X_test_B, batch_size=32)
print ("Loss function w/o B, dim6: ", results_B1)

#originalout_B = np.delete(originalout_A, 4, 1)

del pd_variables[4]
print (" var tolte ", pd_variables)

fig10, axes10 = plt.subplots(nrows=4,ncols=4)
nvar10 = 0
fig10.suptitle("in vs out (SM only) w/o etaj1 & etaj2 & met (dim 6)")
for i in range(nrows):
    for j in range(ncols):
        if nvar10 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes10[i][j].hist(X_test_B[0:,nvar10],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes10[i][j].hist(originalout_B[0:,nvar10],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes10[i][j].hist(out_B1[0:,nvar10],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes10[i][j].set_xlabel(pd_variables[nvar10])            
            nvar10=nvar10+1
fig10.legend(handles=[patch_1, patch_2, patch_3])
fig10.patch.set_facecolor("w")
fig10.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")

############################ etaj1 & etaj2 & met & ptll removed ###########################

vae_C1 = tf.keras.models.load_model("TRAINING/vae_C_denselayers_latentdim6_epoch200_batchsize32_log_eventFiltered")
enc_C1 = tf.keras.models.load_model("TRAINING/enc_C_denselayers_latentdim6_epoch200_batchsize32_log_eventFiltered")

#X_test_C = np.delete(X_test_B, 10, 1)

out_C1 = vae_C1.predict(X_test_C)
results_C1 = vae_C1.evaluate(X_test_C, X_test_C, batch_size=32)
print ("Loss function w/o C (dim6): ", results_C1)

#originalout_C = np.delete(originalout_B, 10, 1)
del pd_variables[10]


print (" var tolte ", pd_variables)

fig11, axes11 = plt.subplots(nrows=4,ncols=4)
nvar11 = 0
fig11.suptitle("in vs out (SM only) w/o etaj1 & etaj2 & met & ptll (dim 6)")
for i in range(nrows):
    for j in range(ncols):
        if nvar11 < len(pd_variables): #range=[-0.1,1.2] da mettere in ogni riemp
            axes11[i][j].hist(X_test_C[0:,nvar11],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="crimson", alpha=0.8)
            axes11[i][j].hist(originalout_C[0:,nvar11],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="midnightblue", alpha=0.8)
            axes11[i][j].hist(out_C1[0:,nvar11],bins=500, range=[-0.1,1.2], weights =wxtest, histtype="step", color="deepskyblue", alpha=0.8)
            axes11[i][j].set_xlabel(pd_variables[nvar11])            
            nvar11=nvar11+1
fig11.legend(handles=[patch_1, patch_2, patch_3])
fig11.patch.set_facecolor("w")
fig11.subplots_adjust(hspace=0.845, right=0.824)
print ("\n\n")








plt.show()



