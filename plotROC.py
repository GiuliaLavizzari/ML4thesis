import tensorflow as tf

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import auc
#from sklearn.metrics import roc_auc_score

cW = 0.3


def effComputation(cW, DIM):
    lossSM = np.loadtxt("lossSM"+str(DIM)+"_noweigths_"+str(cW)+".csv",delimiter=",")
    lossBSM = np.loadtxt("lossBSM"+str(DIM)+"_noweigths_"+str(cW)+".csv",delimiter=",")
    weightsSM= np.loadtxt("weight_SM"+str(DIM)+"_noweigths_"+str(cW)+".csv",delimiter=",")
    weightsBSM= np.loadtxt("weight_BSM"+str(DIM)+"_noweigths_"+str(cW)+".csv",delimiter=",")
    
    effSM = []
    effBSM = []
    for cut in np.arange(0.,0.05,0.00005):
        nSM = 0
        nBSM = 0
        for i in range(len(lossSM)):
            if lossSM[i] > cut:
                nSM = nSM+weightsSM[i]
        for i in range(len(lossBSM)):
            if lossBSM[i] > cut:
                nBSM = nBSM+weightsBSM[i]

        effSM.append(1.*nSM/(weightsSM.sum()))
        effBSM.append(1.*nBSM/(weightsBSM.sum()))
        
    effSMnp = np.array(effSM)
    effBSMnp = np.array(effBSM)
    p = np.argsort(effSMnp)
    effSMs = effSMnp[p]
    effBSMs = effBSMnp[p]
    AUC = auc(effSMs, effBSMs)
    AUC = round(AUC,3)

    return lossSM,lossBSM,weightsSM,weightsBSM,effSM,effBSM,AUC


lossSM_7,lossBSM_7,weightsSM_7,weightsBSM_7,effSM_7,effBSM_7, auc_7 = effComputation(cW, 7)
lossSM_6,lossBSM_6,weightsSM_6,weightsBSM_6,effSM_6,effBSM_6, auc_6 = effComputation(cW, 6)
lossSM_5,lossBSM_5,weightsSM_5,weightsBSM_5,effSM_5,effBSM_5, auc_5 = effComputation(cW, 5)
lossSM_4,lossBSM_4,weightsSM_4,weightsBSM_4,effSM_4,effBSM_4, auc_4 = effComputation(cW, 4)
lossSM_3,lossBSM_3,weightsSM_3,weightsBSM_3,effSM_3,effBSM_3, auc_3 = effComputation(cW, 3)


ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function dim 7")
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(lossBSM_7,bins=100,range=(0.,0.05),weights=weightsBSM_7,histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM Loss")
ax.hist(lossSM_7,bins=100,range=(0.,0.05),weights=weightsSM_7,histtype="step",color="blue",alpha=1.,linewidth=2,density =1, label="SM Loss")
plt.legend(loc=1)
plt.savefig("pROCloss7.png", bbox_inches='tight')
plt.close()

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function dim 6")
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(lossBSM_6,bins=100,range=(0.,0.05),weights=weightsBSM_6,histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM Loss")
ax.hist(lossSM_6,bins=100,range=(0.,0.05),weights=weightsSM_6,histtype="step",color="blue",alpha=1.,linewidth=2,density =1, label="SM Loss")
plt.legend(loc=1)
plt.savefig("pROCloss6.png", bbox_inches='tight')
plt.close()

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function dim 5")
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(lossBSM_5,bins=100,range=(0.,0.05),weights=weightsBSM_5,histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM Loss")
ax.hist(lossSM_5,bins=100,range=(0.,0.05),weights=weightsSM_5,histtype="step",color="blue",alpha=1.,linewidth=2,density =1, label="SM Loss")
plt.legend(loc=1)
plt.savefig("pROCloss5.png", bbox_inches='tight')
plt.close()

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function dim 4")
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(lossBSM_4,bins=100,range=(0.,0.05),weights=weightsBSM_4,histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM Loss")
ax.hist(lossSM_4,bins=100,range=(0.,0.05),weights=weightsSM_4,histtype="step",color="blue",alpha=1.,linewidth=2,density =1, label="SM Loss")
plt.legend(loc=1)
plt.savefig("pROCloss4.png", bbox_inches='tight')
plt.close()

ax = plt.figure(figsize=(7,5), dpi=100, facecolor="w").add_subplot(111)
plt.suptitle("loss function dim 3")
ax.xaxis.grid(True, which="major")
ax.yaxis.grid(True, which="major")
ax.hist(lossBSM_3,bins=100,range=(0.,0.05),weights=weightsBSM_3,histtype="step",color="red",alpha=1.,linewidth=2,density =1,label ="BSM Loss")
ax.hist(lossSM_3,bins=100,range=(0.,0.05),weights=weightsSM_3,histtype="step",color="blue",alpha=1.,linewidth=2,density =1, label="SM Loss")
plt.legend(loc=1)
plt.savefig("pROCloss3.png", bbox_inches='tight')
plt.close()

ax = plt.figure(figsize=(5,5), dpi=100,facecolor="w").add_subplot(111)
plt.suptitle("ROC curves")
ax.xaxis.grid(True, which="minor")
ax.yaxis.grid(True, which="minor")
ax.set_xlim(xmin =0.,xmax=1.1)
ax.set_ylim(ymin =0.,ymax=1.1)
ax.scatter(effSM_7,effBSM_7,color = "blue", s=4, label = "d7 auc="+str(auc_7))
ax.scatter(effSM_6,effBSM_6,color = "green", s=4, label = "d6 auc="+str(auc_6))
ax.scatter(effSM_5,effBSM_5,color = "orange", s=4, label = "d5 auc="+str(auc_5))
ax.scatter(effSM_4,effBSM_4,color = "deepskyblue", s=4, label = "d4 auc="+str(auc_4))
ax.scatter(effSM_3,effBSM_3,color = "mediumvioletred", s=4, label = "d3 auc="+str(auc_3))
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.plot([0.,1],[0.,1],color="r")
plt.xlabel("SM Efficiency")
plt.ylabel("BSM Efficiency")
plt.legend(loc=4)
plt.savefig("pROCcurves.png", bbox_inches='tight')
plt.close()


print ("done")
#plt.show()
