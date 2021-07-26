import tensorflow as tf

import sys
import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import auc
#from sklearn.metrics import roc_auc_score

cW = 0.3
modelN = 1
DIM = 7

def sigmaComputation(cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
                        
    sigma = []
    cut = []
    Signal = []
    Bkg = []
    soverb = []
    
    for k in np.arange(0.,0.08,0.005):
        nS = 0 #signal (lin + quad)
        nB = 0 #background
        k = round(k, 4)
        
        weightsLIN = weightsLIN*cW
        weightsQUAD = weightsQUAD*cW*cW 
                      
        for i in range(len(lossSM)):
            if lossSM[i] > k:
                nB = nB + weightsSM[i]
                print ("\nsm ", weightsSM[i])
        for i in range(len(lossLIN)):
            if lossLIN[i] > k:
            	nS = nS + weightsLIN[i]
            	print ("lin ", weightsLIN[i])
        for i in range(len(lossQUAD)):
            if lossQUAD[i] > k:
            	nS = nS + weightsQUAD[i]
            	print ("quad ", weightsQUAD[i])
        
        if nB == 0.:
            nB = 3.
 
        Signal.append(nS)
        Bkg.append(nB)
        sigma.append(nS/np.sqrt(nB))
        cut.append(k)
        soverb.append(nS/nB)
        
    return sigma, cut, Signal, Bkg, soverb

loss = np.loadtxt("./WandL/TESTlossSM_split.csv", delimiter=",")
lossSM = np.loadtxt("./WandL/TESTlossSM_total.csv", delimiter=",")
lossLIN = np.loadtxt("./WandL/TESTlossLIN_total.csv", delimiter=",")
lossQUAD = np.loadtxt("./WandL/TESTlossQUAD_total.csv", delimiter=",")

weights = np.loadtxt("./WandL/TESTweightsSM_split.csv", delimiter=",")
weightsSM = np.loadtxt("./WandL/TESTweightsSM_total.csv", delimiter=",")
weightsLIN = np.loadtxt("./WandL/TESTweightsLIN_total.csv", delimiter=",")
weightsQUAD = np.loadtxt("./WandL/TESTweightsQUAD_total.csv", delimiter=",")

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function: model "+str(modelN)+", dim "+str(DIM))
ax.set_yscale("log")
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(loss,bins=250,range=[0., 0.08],weights=weights,histtype="step",color="orange",alpha=1.,linewidth=3,density =1,label ="SM test Loss")
ax.hist(lossSM,bins=250,range=[0., 0.08],weights=weightsSM,histtype="step",color="crimson",alpha=.6,linewidth=2,density =1,label ="SM Loss")
ax.hist(lossLIN,bins=250,range=[0., 0.08],weights=weightsLIN*cW,histtype="step",color="blue",alpha=.6,linewidth=2,density =1,label ="LIN Loss")
ax.hist(lossQUAD,bins=250,range=[0., 0.08],weights=weightsQUAD*cW*cW,histtype="step",color="cornflowerblue",alpha=.6,linewidth=2,density =1,label ="QUAD Loss")
plt.legend(loc=1)
ax.patch.set_facecolor("w")
#plt.savefig("./WandL/losses_m"+str(modelN)+"_dim"+str(DIM)+".png", bbox_inches='tight')
#plt.close()

luminosity = 1000.*350. #luminosity expected in 1/pb

cf = 8337.071/np.sum(weights) #first number is the sum of all weights in the original ntuple
cfSM = 8337.071/np.sum(weightsSM)
cfLIN = 251.5229/np.sum(weightsLIN)
cfQUAD = 5835.7227/np.sum(weightsQUAD)

#cf = 22500000./len(weights) #first number is the length of the original ntuple
#cfSM = 22500000./len(weightsSM) 
#cfLIN = 22500000/len(weightsLIN)
#cfQUAD = 22232790/len(weightsQUAD)
print ("cf", cf)
print ("cfSM", cfSM)
print ("cfLIN", cfLIN)
print ("cfQUAD", cfQUAD)

weights = luminosity*weights*cf
weightsSM = luminosity*weightsSM*cfSM
weightsLIN = luminosity*weightsLIN*cfLIN #weights are not yet multiplied by cW
weightsQUAD = luminosity*weightsQUAD*cfQUAD #weights are not yet multiplied by cW

sigma, cut, signal, bkg, soverb = sigmaComputation (0.3, loss, weights, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
sqrtbkg = np.sqrt(bkg)
print (sigma)
print ("\nlen sigma", len(sigma))
print ("\nmax sigma value: ", np.amax(sigma))

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("Signal and Bkg, model "+str(modelN)+" dim "+str(DIM))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(cut, bkg, '-', linewidth = 1.5, color="blue", alpha=1., label="background")
ax.plot(cut, sqrtbkg, '--', linewidth=1.5, color="cornflowerblue", alpha=1., label="sqrt(bkg)")
ax.plot(cut, signal, '-', linewidth = 1.5, color="crimson", alpha=1., label="signal")
ax.set_yscale('log')
ax.set_xlabel("cut on loss function")
plt.legend()
plt.savefig("./WandL/signalandbkgTEST.png", bbox_inches='tight')
#plt.close()


ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("significance, model "+str(modelN)+" dim "+str(DIM))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(cut, sigma, '-', linewidth = 1.5, color="orange", alpha=1.)
ax.set_xlabel("cut on loss function")
ax.set_ylabel("S/sqrt(B)")
plt.savefig("./WandL/significanceTEST.png", bbox_inches='tight')
#plt.close()

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("Signal over Bkg, model "+str(modelN)+" dim "+str(DIM))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(cut, soverb, '-', linewidth = 1.5, color="orange", alpha=1.)
ax.set_xlabel("cut on loss function")
ax.set_ylabel("S/B")
plt.savefig("./WandL/signaloverbkgTEST.png", bbox_inches='tight')


plt.show()
