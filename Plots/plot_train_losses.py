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

mykld = pd.read_csv('./LOSSES/mykld1_training_latentdim7_epoch250_batchsize32_log_eventFiltered.csv')
mykld.columns = ["loss"]
myloss = pd.read_csv('./LOSSES/myloss1_training_latentdim7_epoch250_batchsize32_log_eventFiltered.csv')
myloss.columns = ["loss"]
mymse = pd.read_csv('./LOSSES/mymse1_training_latentdim7_epoch250_batchsize32_log_eventFiltered.csv')
mymse.columns = ["loss"]

mykldD5 = pd.read_csv('./LOSSES/mykld1_training_latentdim5_epoch250_batchsize32_log_eventFiltered.csv')
mykldD5.columns = ["loss"]
mylossD5 = pd.read_csv('./LOSSES/myloss1_training_latentdim5_epoch250_batchsize32_log_eventFiltered.csv')
mylossD5.columns = ["loss"]
mymseD5 = pd.read_csv('./LOSSES/mymse1_training_latentdim5_epoch250_batchsize32_log_eventFiltered.csv')
mymseD5.columns = ["loss"]

mykldD3 = pd.read_csv('./LOSSES/mykld1_training_latentdim3_epoch250_batchsize32_log_eventFiltered.csv')
mykldD3.columns = ["loss"]
mylossD3 = pd.read_csv('./LOSSES/myloss1_training_latentdim3_epoch250_batchsize32_log_eventFiltered.csv')
mylossD3.columns = ["loss"]
mymseD3 = pd.read_csv('./LOSSES/mymse1_training_latentdim3_epoch250_batchsize32_log_eventFiltered.csv')
mymseD3.columns = ["loss"]



#epochs = np.arange(2, 251)


mykldD3 = mykldD3.head(199)
mymseD3 = mymseD3.head(199)
mylossD3 = mylossD3.head(199)
mykldD5 = mykldD5.head(199)
mymseD5 = mymseD5.head(199)
mylossD5 = mylossD5.head(199)
mykld = mykld.head(199)
mymse = mymse.head(199)
myloss = myloss.head(199)


epochs = np.arange(2, 201)

fig, ax = plt.subplots()

ax.plot(epochs, mylossD3["loss"], '-', linewidth = 2.2, color="powderblue", alpha=1., label='loss dim3')
ax.plot(epochs, mykldD3["loss"], '--',linewidth = 1., color="dodgerblue", alpha=1., label='kld dim3')
ax.plot(epochs, mymseD3["loss"], '--',linewidth = 1., color="mediumblue", alpha=1., label='mse dim3')

ax.plot(epochs, mylossD5["loss"], '-', linewidth = 1.5, color="lightgreen", alpha=1., label='loss dim5')
ax.plot(epochs, mykldD5["loss"], '--',linewidth = 1., color="limegreen", alpha=1., label='kld dim5')
ax.plot(epochs, mymseD5["loss"], '--',linewidth = 1., color="seagreen", alpha=1., label='mse dim5')

ax.plot(epochs, myloss["loss"], '-', linewidth = 1.5, color="gold", alpha=1., label='loss dim7')
ax.plot(epochs, mykld["loss"], '--',linewidth = 1., color="orange", alpha=1., label='kld dim7')
ax.plot(epochs, mymse["loss"], '--',linewidth = 1., color="chocolate", alpha=1., label='mse dim7')

ax.set_yscale('log')
ax.set_title('Loss functions (training)')
ax.set_xlabel('epochs')
ax.set_ylabel('loss function')
ax.legend(ncol=3, loc=2)
#ax.legend(loc=2)
fig.subplots_adjust(hspace=0.845, right=0.893, left=0.183)



plt.show()
