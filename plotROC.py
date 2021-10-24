# needs the csv files created by plotCombinedSamples.py

import tensorflow as tf

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import auc
#from sklearn.metrics import roc_auc_score

cW = 0.3


def effComputation(cW, DIM, thr):
    lossSM = np.loadtxt("lossSM"+str(DIM)+"_noweigths_"+str(cW)+".csv",delimiter=",")
    lossBSM = np.loadtxt("lossBSM"+str(DIM)+"_noweigths_"+str(cW)+".csv",delimiter=",")
    weightsSM= np.loadtxt("weight_SM"+str(DIM)+"_noweigths_"+str(cW)+".csv",delimiter=",")
    weightsBSM= np.loadtxt("weight_BSM"+str(DIM)+"_noweigths_"+str(cW)+".csv",delimiter=",")
    
    effSM = []
    effBSM = []
    normSM = 0
    normBSM = 0
    for i in range(len(lossSM)):
    	if lossSM[i] > thr:
    		normSM = normSM+weightsSM[i]
    for i in range(len(lossBSM)):
    	if lossBSM[i] > thr:
    		normBSM = normBSM+weightsBSM[i]
    
    for cut in np.arange(thr,0.05,0.00005):
        nSM = 0
        nBSM = 0
               
        for i in range(len(lossSM)):
            if lossSM[i] > cut:
                nSM = nSM+weightsSM[i]
        for i in range(len(lossBSM)):
            if lossBSM[i] > cut:
                nBSM = nBSM+weightsBSM[i]

        effSM.append(1.*nSM/normSM)
        effBSM.append(1.*nBSM/normBSM)
        
    effSMnp = np.array(effSM)
    effBSMnp = np.array(effBSM)
    p = np.argsort(effSMnp)
    effSMs = effSMnp[p]
    effBSMs = effBSMnp[p]
    AUC = auc(effSMs, effBSMs)
    AUC = round(AUC,3)

    return lossSM,lossBSM,weightsSM,weightsBSM,effSM,effBSM,AUC


lossSM_7,lossBSM_7,weightsSM_7,weightsBSM_7,effSM_7,effBSM_7, auc_7 = effComputation(cW, 7, 0.)
lossSM_6,lossBSM_6,weightsSM_6,weightsBSM_6,effSM_6,effBSM_6, auc_6 = effComputation(cW, 6, 0.)
lossSM_5,lossBSM_5,weightsSM_5,weightsBSM_5,effSM_5,effBSM_5, auc_5 = effComputation(cW, 5, 0.)
lossSM_4,lossBSM_4,weightsSM_4,weightsBSM_4,effSM_4,effBSM_4, auc_4 = effComputation(cW, 4, 0.)
lossSM_3,lossBSM_3,weightsSM_3,weightsBSM_3,effSM_3,effBSM_3, auc_3 = effComputation(cW, 3, 0.)
lossSM_2,lossBSM_2,weightsSM_2,weightsBSM_2,effSM_2,effBSM_2, auc_2 = effComputation(cW, 2, 0.)


ax = plt.figure(figsize=(5,5), dpi=100,facecolor="w").add_subplot(111)
plt.suptitle("ROC curves")
ax.xaxis.grid(True, which="minor")
ax.yaxis.grid(True, which="minor")
ax.set_xlim(xmin =0.,xmax=1.1)
ax.set_ylim(ymin =0.,ymax=1.1)
ax.scatter(effSM_7,effBSM_7,color = "blue", s=4, alpha = 0.6, label = "d7 thr=0.007 auc="+str(auc_7))
ax.scatter(effSM_6,effBSM_6,color = "green", s=4, alpha = 0.6, label = "d6 thr=0.010 auc="+str(auc_6))
ax.scatter(effSM_5,effBSM_5,color = "orange", s=4, alpha = 0.6, label = "d5 thr=0.017 auc="+str(auc_5))
ax.scatter(effSM_4,effBSM_4,color = "deepskyblue", s=4, alpha = 0.6, label = "d4 thr=0.012 auc="+str(auc_4))
ax.scatter(effSM_3,effBSM_3,color = "mediumvioletred", s=4, alpha = 0.6, label = "d3 thr=0.018 auc="+str(auc_3))
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.plot([0.,1],[0.,1],color="r")
plt.xlabel("SM Efficiency")
plt.ylabel("BSM Efficiency")
plt.legend(loc=4)
plt.savefig("pROCcurves_cW03.png", bbox_inches='tight')
plt.close()


print ("done")
#plt.show()
