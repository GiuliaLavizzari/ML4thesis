# INDEX
#
# uploading losses and weights
# normalization
# plotting lf
# plotting lf with root
# plots (signal&bkg, significance)
# cW sensibility
# bisection


import tensorflow as tf

import ROOT
import sys
import numpy as np
import matplotlib
from array import array
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import auc
#from sklearn.metrics import roc_auc_score

#cW = 0.3
modelN = 1
DIM = 7


'''
def sigmaComputation(start, stop, step, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
                        
    sigma = []
    cut = []
    Signal = []
    Bkg = []
    
    weightsLIN = weightsLIN*cW
    weightsQUAD = weightsQUAD*cW*cW 
    
    for k in np.arange(start,stop,step):
        nS = 0. #signal (lin + quad)
        nB = 0. #background
                      
        for i in range(len(lossSM)):
            if lossSM[i] > k:
                nB = nB + weightsSM[i]
        for i in range(len(lossLIN)):
            if lossLIN[i] > k:
                nS = nS + weightsLIN[i]
        for i in range(len(lossQUAD)):
            if lossQUAD[i] > k:
                nS = nS + weightsQUAD[i]
        
        if nB >= 0.001:
            Signal.append(nS)
            Bkg.append(nB)
            sigma.append(nS/np.sqrt(nB))
            cut.append(k)
        
    return sigma, cut, Signal, Bkg
'''

def sigmaFunction(k, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
    nS = 0. #signal (lin + quad)
    nB = 0. #background
                      
    for i in range(len(lossSM)):
        if lossSM[i] > k:
            nB = nB + weightsSM[i]
    for i in range(len(lossLIN)):
        if lossLIN[i] > k:
            nS = nS + weightsLIN[i]*cW
    for i in range(len(lossQUAD)):
        if lossQUAD[i] > k:
            nS = nS + weightsQUAD[i]*cW*cW
    
    if nB == 0.:
        nB = 0.5
        nS = 0.
    sigma = nS/np.sqrt(nB)
    return nS, nB, sigma


def sigmaComputation(start, stop, step, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
                        
    Sigma = []
    Cut = []
    Signal = []
    Bkg = []
    
    for k in np.arange(start,stop,step):
        k = round(k, 4)
        ns, nb, sig = sigmaFunction(k, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        
        if nb >= 1.: 
            Signal.append(ns)
            Bkg.append(nb)
            Sigma.append(sig)
            Cut.append(k)
        
    return Sigma, Cut, Signal, Bkg    

# if the maximum of the sigma(cW) is below 3, the corresponding cW value is considered not sensible to BSM data
   
def cWsensibility(start, stop, step, cWstart, cWstop, cWstep, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
    sensible = []
    notsensible = []
    
    cWarr = np.arange(cWstart, cWstop, cWstep)
    np.around(cWarr, 4)
    print (cWarr)
    
    for i in range(len(cWarr)):
        sigma,_,_,_ = sigmaComputation(start, stop, step, cWarr[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        maxsigma = np.amax(sigma)
        if maxsigma < 3.:
            notsensible.append(cWarr[i])
        if maxsigma >= 3. :
            sensible.append(cWarr[i])
            print ("per cW ", round(cWarr[i], 4), " abbiamo max pari a ", maxsigma)
    print (len(notsensible)-1)
    lastNS = notsensible[len(notsensible)-1]
    firstS = sensible[0]
    return lastNS, firstS
    

def bisectionINCR(idx, cut, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
    
    A = cut[0]
    B = cut[idx]    
    _,_,sigma_A = sigmaFunction(A, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    _,_,sigma_B = sigmaFunction(B, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    
    if sigma_A > 3. :
        print ("no sigma in this interval - incr")
        return 5., 5. # control value (arbitrary, must be big enough not to be mistaken for some true value of loss cut)
    
    elif sigma_A == 3. :
        print ("exact solution")
        return A, 0.
    
    elif sigma_B == 3. :
    	print ("exact solution")
    	return B, 0.
    
    else :
        C = (A+B)/2
        _,_,sigma_C = sigmaFunction(C, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        i = 1. # with i==1 while cycle is running
        while i == 1. :
            if sigma_C >= 3. :
                B = C
            if sigma_C < 3. :
                A = C
            C = (A+B)/2
            _,_,sigma_C = sigmaFunction(C, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
            
            control = abs(sigma_C - 3.)
            if control > 0.0001 :
                i = 1.
            else :
                if sigma_C >= 3. :
                    i = 0.
                else :
                    i = 0.
        errC = (B-A)/2
        return C, errC            
   
def bisectionDECR(idx1, idx2, cut, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
    
    A = cut[idx1]
    B = cut[idx2]    
    _,_,sigma_A = sigmaFunction(A, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    _,_,sigma_B = sigmaFunction(B, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    
    if sigma_B > 3. :
        print ("no sigma in this interval - decr")
        return 5., 5. # control value (arbitrary, must be big enough not to be mistaken for some true value of loss cut)
    
    elif sigma_A == 3. :
        print ("exact solution")
        return A, 0.
    
    elif sigma_B == 3. :
    	print ("exact solution")
    	return B, 0.
    
    else :
        C = (A+B)/2
        _,_,sigma_C = sigmaFunction(C, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        i = 1. # with i==1 while cycle is running
        while i == 1. :
            if sigma_C < 3. :
                B = C
            if sigma_C >= 3. :
                A = C
            C = (A+B)/2
            _,_,sigma_C = sigmaFunction(C, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
            
            control = abs(sigma_C - 3.)
            if control > 0.0001 :
                i = 1.
            else :
                if sigma_C >= 3. :
                    i = 0.
                else :
                    i = 0.
        errC = (B-A)/2
        return C, errC            



############### uploading losses and weights ###############

lossSM = np.loadtxt("./WandL/TESTlossSM_split.csv", delimiter=",") #test 
lossLIN = np.loadtxt("./WandL/TESTlossLIN_total.csv", delimiter=",")
lossQUAD = np.loadtxt("./WandL/TESTlossQUAD_total.csv", delimiter=",")

weightsSM = np.loadtxt("./WandL/TESTweightsSM_split.csv", delimiter=",") #test
weightsLIN = np.loadtxt("./WandL/TESTweightsLIN_total.csv", delimiter=",")
weightsQUAD = np.loadtxt("./WandL/TESTweightsQUAD_total.csv", delimiter=",")


##################### normalization #########################

luminosity = 1000.*350. #luminosity expected in 1/pb

fSM = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
hSM = fSM.Get("SSWW_SM_nums")
xsecSM = hSM.GetBinContent(1)
sumwSM = hSM.GetBinContent(2)
normSM = 5.* xsecSM * luminosity / (sumwSM) # on test set (0.2*total)

fLIN = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuple_SSWW_cW_LI.root")
hLIN = fLIN.Get("SSWW_cW_LI_nums")
xsecLIN = hLIN.GetBinContent(1)
sumwLIN = hLIN.GetBinContent(2)
normLIN = xsecLIN * luminosity / (sumwLIN)

fQUAD = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuple_SSWW_cW_QU.root")
hQUAD = fQUAD.Get("SSWW_cW_QU_nums")
xsecQUAD = hQUAD.GetBinContent(1)
sumwQUAD = hQUAD.GetBinContent(2)
normQUAD = xsecQUAD * luminosity / (sumwQUAD)

print ("normSM", normSM)
print ("normLIN", normLIN)
print ("normQUAD", normQUAD)

weightsSM = weightsSM*normSM
weightsLIN = weightsLIN*normLIN # weights are not yet multiplied by cW
weightsQUAD = weightsQUAD*normQUAD # weights are not yet multiplied by cW*cW

###################### plotting lf #########################
'''
wLIN3 = weightsLIN*0.3
wQUAD3 = weightsQUAD*0.3*0.3
weightsBSM3 = np.concatenate((wLIN3, wQUAD3), axis=0)

wLIN5 = weightsLIN*0.5
wQUAD5 = weightsQUAD*0.5*0.5
weightsBSM5 = np.concatenate((wLIN5, wQUAD5), axis=0)

wLIN1 = weightsLIN*0.1
wQUAD1 = weightsQUAD*0.1*0.1
weightsBSM1 = np.concatenate((wLIN1, wQUAD1), axis=0)

wLIN9 = weightsLIN*0.9
wQUAD9 = weightsQUAD*0.9*0.9
weightsBSM9 = np.concatenate((wLIN9, wQUAD9), axis=0)

lossBSM = np.concatenate((lossLIN, lossQUAD), axis=0)

lossALL = np.concatenate((lossSM, lossBSM), axis=0)
weightsALL1 = np.concatenate((weightsSM, weightsBSM1), axis=0)
weightsALL3 = np.concatenate((weightsSM, weightsBSM3), axis=0)
weightsALL5 = np.concatenate((weightsSM, weightsBSM5), axis=0)

#print ("\nnumpy weights ", np.sum(wSMtest))

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function: model "+str(modelN)+", dim "+str(DIM))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.hist(lossALL,bins=150,range=[0., 0.05], weights=weightsALL,histtype="step",color="blue",alpha=.6,linewidth=2,label ="ALL Loss cW=0.3")
ax.hist(lossSM,bins=150,range=[0., 0.05], weights=weightsSM,histtype="step",color="crimson",alpha=.6,linewidth=2,label ="SM test Loss")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL5,histtype="step",color="blue",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.5")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL3,histtype="step",color="dodgerblue",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.3")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL1,histtype="step",color="cyan",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.1")
ax.hist(lossBSM,bins=150,range=[0., 0.05],weights=weightsBSM9,histtype="step",color="midnightblue",alpha=.6,linewidth=2,label ="BSM Loss cW=0.9")
ax.hist(lossBSM,bins=150,range=[0., 0.05],weights=weightsBSM5,histtype="step",color="blue",alpha=.6,linewidth=2,label ="BSM Loss cW=0.5")
ax.hist(lossBSM,bins=150,range=[0., 0.05],weights=weightsBSM3,histtype="step",color="dodgerblue",alpha=.6,linewidth=2,label ="BSM Loss cW=0.3")
ax.hist(lossBSM,bins=150,range=[0., 0.05],weights=weightsBSM1,histtype="step",color="cyan",alpha=.6,linewidth=2,label ="BSM Loss cW=0.1")
ax.set_yscale("log")
plt.legend(loc=1)
ax.patch.set_facecolor("w")
plt.savefig("./WandL/lossesFINAL_m"+str(modelN)+"_dim"+str(DIM)+".png", bbox_inches='tight')
plt.close()
'''

################ plotting lfSM with root ####################
'''
h = ROOT.TH1D("h_sm", "h_sm", 150, 0., 0.05)
h.SetLineColor(ROOT.kRed)
h.FillN(len(loss), array('d', lossSM), array('d', weightsSM) )
h.Scale(norm)

ROOT.gStyle.SetOptStat(0)
leg = ROOT.TLegend(0.89, 0.89, 0.7, 0.7)
leg.SetBorderSize(0)
leg.AddEntry(h, "350fb^{-1}", "F")
c  = ROOT.TCanvas("c", "c", 1000, 1000)

h.Draw("hist")
c.SetLogy()
leg.Draw()
c.Draw()

print("\nRoot integral: ",h.Integral())
'''

######################### plots ################################
'''

cWarr = np.arange(0.1, 1., 0.1)
np.around(cWarr, 4)
for i in range((len(cWarr)-1)):
    sigma, cut, signal, bkg = sigmaComputation(0., 0.04, 0.001, cWarr[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    print ("\n",sigma)
    print ("\n",cut)
    print ("\n",signal)
    print ("\n",bkg)
    
    sqrtbkg = np.sqrt(bkg)
    ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
    plt.suptitle("Signal and Bkg, model "+str(modelN)+" cW "+str(round(cWarr[i],2)))
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    ax.plot(cut, bkg, '-', linewidth = 1.5, color="blue", alpha=1., label="background")
    ax.plot(cut, sqrtbkg, '--', linewidth=1.5, color="cornflowerblue", alpha=1., label="sqrt(bkg)")
    ax.plot(cut, signal, '-', linewidth = 1.5, color="crimson", alpha=1., label="signal")
    #ax.set_yscale('log')
    ax.set_xlabel("cut on loss function")
    plt.legend()
    plt.savefig("./WandL/signalandbkg"+str(modelN)+"_cW"+str(round(cWarr[i],2))+".png", bbox_inches='tight')
    plt.close()
    
    horizontal_line = np.array([3 for h in range(len(cut))])
    ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
    plt.suptitle("Significance, model "+str(modelN)+" cW "+str(round(cWarr[i],2)))
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    ax.plot(cut, sigma, '-', linewidth = 1.5, color="orange", alpha=1.)
    ax.set_xlabel("cut on loss function")
    ax.set_ylabel("S/sqrt(B)")
    plt.plot(cut,horizontal_line,"--",color="r")
    plt.savefig("./WandL/significance"+str(modelN)+"_cW"+str(round(cWarr[i],2))+".png", bbox_inches='tight')
    plt.close()
 '''   

################## cW sensibility #############################
'''
# cWsensibility (start, stop, step, cWstart, cWstop, cWstep)
Lns1, Fs1 = cWsensibility(0., 0.04, 0.001, 0., 1., 0.1, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
print ("loss of sensibility within ", Lns1, " - ", Fs1)
Lns2, Fs2 = cWsensibility(0., 0.04, 0.001, round(Lns1, 4), round(Fs1, 4), 0.01, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
print ("loss of sensibility within ", Lns2, " - ", Fs2)
Lns3, Fs3 = cWsensibility(0., 0.04, 0.001, round(Lns2, 4), round(Fs2, 4), 0.001, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
print ("loss of sensibility within ", Lns3, " - ", Fs3)
#Lns4, Fs4 = cWsensibility(0., 0.4, 0.1, round(Lns3, 4), round(Fs3, 4), 0.0001, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
#print ("loss of sensibility within ", Lns4, " - ", Fs4)

# Fs3 is the first sensible value, 0.001 step
'''
##################################### 

Fs2 = 0.35
#cWcomp = np.arange(round(Fs3, 3), 0.6, 0.001)
#cWcomp = np.around(cWcomp, decimals = 3)

cWcomp = np.arange(round(Fs2, 2), 0.54, 0.01)
print (cWcomp)
cWcomp = np.around(cWcomp, decimals = 2)

cW1 = [] # monotonic increasing
cut1 = []
err1 = []
cW2 = [] # monotonic decreasing
cut2 = []
err2 = []

for i in range(len(cWcomp)):
    sigma,cut,_,_ = sigmaComputation(0., 0.04, 0.001, cWcomp[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    print ("\ncW ", cWcomp[i])
    idx = np.argmax(sigma)
    idxfin = len(cut)-1
    val1, error1 = bisectionINCR(idx, cut, cWcomp[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    val2, error2 = bisectionDECR(idx, idxfin, cut, cWcomp[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    
    if val1 != 5. :
        cW1.append(cWcomp[i])
        cut1.append(val1)
        err1.append(error1)
    
    if val2 != 5. :
        cW2.append(cWcomp[i])
        cut2.append(val2)
        err2.append(error2)
    print ("finito")


print ("\nMonotonic increasing: ")
for i in range(len(cW1)):
    print ("cW:", cW1[i], " - k =", cut1[i], " +- ", err1[i])
    
print ("\nMonotonic decreasing: ")
for i in range(len(cW2)):
    print ("cW:", cW2[i], " - k =", cut2[i], " +- ", err2[i])


ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss cut (:sigma==3) as a function of cW, model "+str(modelN))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(cW1, cut1, '--', linewidth = 1.5, color="mediumslateblue", alpha=0.8, label="branch 1 (incr)")
ax.plot(cW2, cut2, '--', linewidth = 1.5, color="coral", alpha=0.8, label="branch 2 (decr)")
ax.set_xlabel("cW")
ax.set_ylabel("cut")
plt.legend(loc=4)
plt.savefig("./WandL/cutsandcW"+str(modelN)+".png", bbox_inches='tight')
#plt.close()


   
print ("done")
plt.show()
