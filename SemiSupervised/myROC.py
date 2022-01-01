from sklearn import datasets
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


#taking the model
#from VAE_model_extended_moreDKL import *
from VAE_testReturnLoss import *
from matplotlib import pyplot as plt

import os

import ROOT

def TFposRate (opM, opT):
    intermediate_dim = 20 #50 by default
    input_dim = 10 #was 20 in default
    half_input = 7 #was 20 in the newTest
    latent_dim = 3 #was 3 for optimal performance
    epochs = 200 #100
    batchsize=64 #32
    nameExtenstion = str(opM)+"_"+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs)+"_"+str(batchsize)
    filename = "./storage/posRate_model"+str(nameExtenstion)+"_tested"+str(opT)+".txt"
    filename2 = "./storage/plot_model"+str(nameExtenstion)+"_tested"+str(opT)+".txt"
    if os.path.isfile(filename) == True:
        print ("file already present")
        FPR, TPR = np.loadtxt(filename, unpack=True)
        return FPR, TPR
    else :
        print ("file not present: computing...")
        pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
        kinematicFilter = "ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200"
        dfSM = ROOT.RDataFrame("SSWW_SM","../ntuplesBSM/ntuple_SSWW_SM.root")
        dfSM = dfSM.Filter(kinematicFilter)
        dfBSM = ROOT.RDataFrame("SSWW_"+str(opT)+"_QU","../ntuplesBSM/ntuple_SSWW_"+str(opT)+"_QU.root")
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
            npd[vars] = npd[vars].apply(np.log10)
            npd_BSM[vars] = npd_BSM[vars].apply(np.log10)
        # "labelling SM (0) and BSM (1)"
        Y_true = np.full(nEntries,0)
        Y_true_BSM = np.full(nEntries,1)
        X_train, X_test, y_train, y_test = train_test_split(npd, Y_true, test_size=0.2, random_state=1)
        wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
        BSM_train, BSM_test, y_BSM_train, y_BSM_test = train_test_split(npd_BSM, Y_true_BSM, test_size=0.2, random_state=1)
        wBSM_train, wBSM_test, _ , _ = train_test_split(wpdBSM, wpdBSM, test_size=0.2, random_state=1)
        wx = wx_train["w"].to_numpy()
        wxtest = wx_test["w"].to_numpy()
        wBSM = wBSM_train["w"].to_numpy()
        wBSMtest = wBSM_test["w"].to_numpy()
        # scale data
        t = MinMaxScaler()
        #t = StandardScaler()
        t.fit(X_train)
        X_test = t.transform(X_test)
        BSM_test = t.transform(BSM_test)
        n_inputs = npd.shape[1]
        original_dim = n_inputs
        vae = tf.keras.models.load_model('vae_test_newModelUsingLatentSpace_'+nameExtenstion)
        #concatenating SM and BSM
        label_test = np.concatenate((y_test,y_BSM_test))
        weights_test=np.concatenate((wxtest,wBSMtest))
        output_SM = vae.predict(X_test)
        output_BSM = vae.predict(BSM_test)
        output = np.concatenate((output_SM,output_BSM))
        FPR, TPR, threshold1 = roc_curve(label_test, output)
        np.savetxt(filename, np.column_stack([FPR, TPR]))
        np.savetxt(filename2, np.column_stack([output_SM, wxtest, output_BSM, wBSMtest]) )
        return FPR, TPR


def TFposRatebeta (opM, beta, opT):
    intermediate_dim = 20 #50 by default
    input_dim = 10 #was 20 in default
    half_input = 7 #was 20 in the newTest
    latent_dim = 3 #was 3 for optimal performance
    epochs = 200 #100
    batchsize=64 #32
    nameExtenstion = str(opM)+"_beta"+str(beta)+"_"+str(intermediate_dim)+"_"+str(input_dim)+"_"+str(half_input)+"_"+str(latent_dim)+"_"+str(epochs)+"_"+str(batchsize)
    filename = "./storage/posRate_model"+str(nameExtenstion)+"_tested"+str(opT)+".txt"
    filename2 = "./storage/plot_model"+str(nameExtenstion)+"_tested"+str(opT)+".txt"
    if os.path.isfile(filename) == True:
        print ("file already present")
        FPR, TPR = np.loadtxt(filename, unpack=True)
        return FPR, TPR
    else :
        print ("file not present: computing...")
        pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
        kinematicFilter = "ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200"
        dfSM = ROOT.RDataFrame("SSWW_SM","../ntuplesBSM/ntuple_SSWW_SM.root")
        dfSM = dfSM.Filter(kinematicFilter)
        dfBSM = ROOT.RDataFrame("SSWW_"+str(opT)+"_QU","../ntuplesBSM/ntuple_SSWW_"+str(opT)+"_QU.root")
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
            npd[vars] = npd[vars].apply(np.log10)
            npd_BSM[vars] = npd_BSM[vars].apply(np.log10)
        # "labelling SM (0) and BSM (1)"
        Y_true = np.full(nEntries,0)
        Y_true_BSM = np.full(nEntries,1)
        X_train, X_test, y_train, y_test = train_test_split(npd, Y_true, test_size=0.2, random_state=1)
        wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
        BSM_train, BSM_test, y_BSM_train, y_BSM_test = train_test_split(npd_BSM, Y_true_BSM, test_size=0.2, random_state=1)
        wBSM_train, wBSM_test, _ , _ = train_test_split(wpdBSM, wpdBSM, test_size=0.2, random_state=1)
        wx = wx_train["w"].to_numpy()
        wxtest = wx_test["w"].to_numpy()
        wBSM = wBSM_train["w"].to_numpy()
        wBSMtest = wBSM_test["w"].to_numpy()
        # scale data
        t = MinMaxScaler()
        #t = StandardScaler()
        t.fit(X_train)
        X_test = t.transform(X_test)
        BSM_test = t.transform(BSM_test)
        n_inputs = npd.shape[1]
        original_dim = n_inputs
        vae = tf.keras.models.load_model('vae_test_newModelUsingLatentSpace_'+nameExtenstion)
        #concatenating SM and BSM
        label_test = np.concatenate((y_test,y_BSM_test))
        weights_test=np.concatenate((wxtest,wBSMtest))
        output_SM = vae.predict(X_test)
        output_BSM = vae.predict(BSM_test)
        output = np.concatenate((output_SM,output_BSM))
        FPR, TPR, threshold1 = roc_curve(label_test, output)
        np.savetxt(filename, np.column_stack([FPR, TPR]))
        np.savetxt(filename2, np.column_stack([output_SM, wxtest, output_BSM, wBSMtest]) )
        return FPR, TPR

falsepos1, truepos1 = TFposRate("cW", "cW")
falsepos12, truepos12 = TFposRate("cW", "cHWB") # piuttosto scarso
falsepos13, truepos13 = TFposRate("cW", "cll1") # scarso (diago)
falsepos14, truepos14 = TFposRate("cW", "cqq3")

falsepos2, truepos2 = TFposRate("cHWB", "cHWB") # non scarso me lo sarei aspettato peggio
falsepos22, truepos22 = TFposRate("cHWB", "cW") # sopra diago
falsepos23, truepos23 = TFposRate("cHWB", "cll1") # scarso (diago)
falsepos24, truepos24 = TFposRate("cHWB", "cqq3") # addirttura sotto diago

falsepos3, truepos3 = TFposRate("cll1", "cll1") # scarso (diago)
falsepos32, truepos32 = TFposRate("cll1", "cW") 
falsepos33, truepos33 = TFposRate("cll1", "cHWB") 
falsepos34, truepos34 = TFposRate("cll1", "cqq3")

falsepos4, truepos4 = TFposRate("cqq3", "cqq3")
falsepos5, truepos5 = TFposRate("cqq3", "cqq3")

ax = plt.figure(figsize=(7,5), dpi=100,facecolor="w").add_subplot(111)
plt.suptitle('Receiver Operating Characteristic - VAE')
ax.plot(falsepos1, truepos1,"-",color="royalblue",label="_cW_QU")
#ax.plot(falsepos12, truepos12,"--",color="royalblue",label="_cW_QU")
#ax.plot(falsepos13, truepos13,",",color="royalblue",label="_cW_QU")
#ax.plot(falsepos14, truepos14,"-.",color="royalblue",label="_cW_QU")
ax.plot(falsepos2, truepos2,"-",color="gold",label="_cW_QU")
#ax.plot(falsepos22, truepos22,"--",color="gold",label="_cW_QU")
#ax.plot(falsepos23, truepos23,",",color="gold",label="_cW_QU")
#ax.plot(falsepos24, truepos24,"-.",color="gold",label="_cW_QU")
ax.plot(falsepos3, truepos3,"-",color="chartreuse",label="_cW_QU")
#ax.plot(falsepos32, truepos32,"--",color="chartreuse",label="_cW_QU")
#ax.plot(falsepos33, truepos33,",",color="chartreuse",label="_cW_QU")
#ax.plot(falsepos34, truepos34,"-.",color="chartreuse",label="_cW_QU")
ax.plot(falsepos4, truepos4,"-",color="crimson",label="_cW_QU")
#ax.plot(falsepos5, truepos5,"--",color="orange",label="_cW_QU")
ax.plot([0, 1], ls="--")
ax.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()



falseposB1, trueposB1 = TFposRatebeta("cW", 0.001, "cW")
falseposB2, trueposB2 = TFposRatebeta("cW", 1e-11, "cW")

ax = plt.figure(figsize=(7,5), dpi=100,facecolor="w").add_subplot(111)
plt.suptitle('Receiver Operating Characteristic - VAE')
ax.plot(falsepos1, truepos1,"-",color="orange",label="_cW_QU")
ax.plot(falseposB1, trueposB1,"-.",color="crimson",label="_cW_QU")
ax.plot(falseposB2, trueposB2,"-",color="gold",label="_cW_QU")
ax.plot([0, 1], ls="--")
ax.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()


'''
#
# variable from the nutple
#







#print('roc_auc_score for VAE_cW_QU: ', roc_auc_score(label_test,output))


'''

'''

bins=100
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(output_SM,bins=bins, weights=wxtest,density=1,range=[0.,1.],histtype="step",color="red",alpha=0.6,linewidth=2,label=nameExtenstion+"_SM Output")                        
ax.hist(output_BSM,bins=bins, weights=wBSMtest, density=1,range=[0.,1.],histtype="step",color="blue",alpha=0.6,linewidth=2,label=nameExtenstion+"_cW_QU Output")                        
ax.hist(output_BSM2,bins=bins, weights=wBSM2test,density=1,range=[0.,1.],histtype="step",color="green",alpha=0.6,linewidth=2,label=nameExtenstion+"_cqq3 Output")                        

ax.hist(output_vae2_SM,bins=bins, weights=wxtest,density=1,range=[0.,1.],histtype="step",color="orange",alpha=0.6,linewidth=2,label=nameExtenstion2+"_SM Output")                        
ax.hist(output_vae2_BSM,bins=bins, weights=wBSMtest, density=1,range=[0.,1.],histtype="step",color="black",alpha=0.6,linewidth=2,label=nameExtenstion2+"_cW_QU Output")                        
ax.hist(output_vae2_BSM2,bins=bins, weights=wBSM2test,density=1,range=[0.,1.],histtype="step",color="purple",alpha=0.6,linewidth=2,label=nameExtenstion2+"_cqq3 Output")                        
plt.legend()  

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - VAE')
plt.plot(false_positive_rate1, true_positive_rate1,label=nameExtenstion+"_cW_QU")
plt.plot(false_positive_rate2, true_positive_rate2,label=nameExtenstion+"_cqq3_QU")
plt.plot(false_positive_rate1_vae2, true_positive_rate1_vae2,label=nameExtenstion2+"_cW_QU")
plt.plot(false_positive_rate2_vae2, true_positive_rate2_vae2,label=nameExtenstion2+"_cqq3_QU")
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()


'''


plt.show()
