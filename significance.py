import tensorflow as tf

import ROOT
import sys
import numpy as np
import matplotlib
from array import array
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import auc
#from sklearn.metrics import roc_auc_score

#cW = 0.3
modelN = 333
DIM = 3

                     

def sigmaFunction(k, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
# returns the value of sigma at a given cW and k

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
        nB = 0.5 # arbitrary value lower than 1.
        nS = 0.
    sigma = nS/np.sqrt(nB)
    return nS, nB, sigma


def sigmaComputation(start, stop, step, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
#returns the lists of values of sigma(cut) and cut such that nB > 1. (selects the cuts for which the number of SM events is at least 1)                     
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
    

def GoldenRatioSearch (a, b, err, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD):
    # returns the MAXIMUM VALUE of sigma(k)
    r = (np.sqrt(5)-1)/2
    a1 = b - r*(b-a)
    a2 = a + r*(b-a)
    
    while abs(b-a) > err :
        _,_,f1 = sigmaFunction(a1, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        _,_,f2 = sigmaFunction(a2, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
        
        if f1 < f2 : # > for minimum
            a = a1
            a1 = a2
            f1 = f2
            a2 = a + r*(b-a)
        else :
            b = a2
            a2 = a1
            f2 = f1
            a1 = b - r*(b-a)
        
    x_opt = (a + b)/2
    _,_,f_opt = sigmaFunction(x_opt, cW, lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    
    return x_opt, f_opt, err 


# FOLLOWING PARTS:
# uploading losses and weights
# normalization
# plotting lf
# plotting lf with root
# plots (signal&bkg, significance)
# cW sensibility
# maximum sigma value



############### uploading losses and weights ###############

lossSM = np.loadtxt("./WandL/model333S_lossSM_split.csv", delimiter=",") #computed only on test set
lossLIN = np.loadtxt("./WandL/model333S_lossLIN_total.csv", delimiter=",")
lossQUAD = np.loadtxt("./WandL/model333S_lossQUAD_total.csv", delimiter=",")

#weights are not scaled by cW yet
weightsSM = np.loadtxt("./WandL/model333S_weightsSM_split.csv", delimiter=",") #only test set
weightsLIN = np.loadtxt("./WandL/model333S_weightsLIN_total.csv", delimiter=",")
weightsQUAD = np.loadtxt("./WandL/model333S_weightsQUAD_total.csv", delimiter=",")


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

lossmax = np.amax(lossBSM)
print  (lossmax)

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function: model "+str(modelN)+", dim "+str(DIM))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
#ax.hist(lossALL,bins=150,range=[0., 0.05], weights=weightsALL,histtype="step",color="blue",alpha=.6,linewidth=2,label ="ALL Loss cW=0.3")
ax.hist(lossSM,bins=150,range=[0., lossmax], weights=weightsSM,histtype="step",color="crimson",alpha=.6,linewidth=2,label ="SM test Loss")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL5,histtype="step",color="blue",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.5")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL3,histtype="step",color="dodgerblue",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.3")
#ax.hist(lossALL,bins=150,range=[0., 0.05],weights=weightsALL1,histtype="step",color="cyan",alpha=.6,linewidth=2,label ="Loss SM+BSM cW=0.1")
ax.hist(lossBSM,bins=150,range=[0., lossmax],weights=weightsBSM9,histtype="step",color="midnightblue",alpha=.6,linewidth=2,label ="BSM Loss cW=0.9")
ax.hist(lossBSM,bins=150,range=[0., lossmax],weights=weightsBSM5,histtype="step",color="blue",alpha=.6,linewidth=2,label ="BSM Loss cW=0.5")
ax.hist(lossBSM,bins=150,range=[0., lossmax],weights=weightsBSM3,histtype="step",color="dodgerblue",alpha=.6,linewidth=2,label ="BSM Loss cW=0.3")
ax.hist(lossBSM,bins=150,range=[0., lossmax],weights=weightsBSM1,histtype="step",color="cyan",alpha=.6,linewidth=2,label ="BSM Loss cW=0.1")
ax.set_yscale("log")
plt.legend(loc=1)
ax.patch.set_facecolor("w")
plt.savefig("./WandL/lossesFINAL_m"+str(modelN)+"_dim"+str(DIM)+".png", bbox_inches='tight')
#plt.close()


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

cWarr = np.arange(0.1, 1., 0.1)
np.around(cWarr, 4)
for i in range((len(cWarr)-1)):
    sigma, cut, signal, bkg = sigmaComputation(0., round(lossmax, 2), 0.005, cWarr[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
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

################## cW sensibility #############################

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


################## max sigma value ################### 

cWcomp = np.arange(0.1, 1., 0.05)
print (cWcomp)
cWcomp = np.around(cWcomp, decimals = 2)

maximum = [] 

for i in range(len(cWcomp)):
    sigma,cut,_,_ = sigmaComputation(0., round(lossmax, 2), 0.005, cWcomp[i], lossSM, weightsSM, lossLIN, weightsLIN, lossQUAD, weightsQUAD)
    print ("\ncW ", cWcomp[i])
    cutmax, sigmamax, errsigmamax = GoldenRatioSearch(cut[0], cut[len(cut)-1], 0.00001, cWcomp[i], lossSM, weightsSM, lossLIN, weightsLIN, 								lossQUAD, weightsQUAD)
    #print ("maximum sigma value is ", sigmamax, " +- nonperv", "  at cut = ", cutmax)
    maximum.append(sigmamax)
    print ("finito")

print ("\ncut")
print (cWcomp)
print ("\nsigma")
print (maximum)


horizontal_line = np.array([3 for h in range(len(cWcomp))])
ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("maximum sigma value as a function of cW, model "+str(modelN))
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.plot(cWcomp, maximum, '-', linewidth = 1.5, color="green", alpha=1., label="model 1")
plt.plot(cWcomp,horizontal_line,"--",color="r")
#plt.legend(loc=2)
ax.set_xlabel("cW")
ax.set_ylabel("sigma max")
plt.subplots_adjust(bottom=0.145)
plt.xticks(np.arange(0.1, 1., 0.05), rotation=50)
plt.savefig("./WandL/maximumsigma"+str(modelN)+".png", bbox_inches='tight')
plt.close()


   
print ("done")
#plt.show()
