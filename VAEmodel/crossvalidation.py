import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import ROOT
import sys
import numpy as np
import pandas as pd

#taking the model
from finalVAE1 import *
MODEL = 1

#ROOT.ROOT.EnableImplicitMT()

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']

# cuts
dfAll = ROOT.RDataFrame("SSWW_SM","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")

npy = df.AsNumpy(pd_variables)
wSM = df.AsNumpy("w")
npd =pd.DataFrame.from_dict(npy)
wpdSM = pd.DataFrame.from_dict(wSM)

nEntries = 920760
npd = npd.head(nEntries)
wpdSM = wpdSM.head(nEntries)

# log
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1', 'ptl2', 'ptll']:
	npd[vars] = npd[vars].apply(np.log10)


#VAE params:
DIM = 7#sys.argv[1] #latent dimension
BATCH = 32 #batch size
EPOCHS = 250 #number of epochs
#EPOCHS = int(EPOCHS)

n_inputs = npd.shape[1] 
original_dim = n_inputs

vae = VariationalAutoEncoder(original_dim, DIM) # (self, original, latent)
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=tf.keras.losses.MeanSquaredError())

#Implementing cross validation
k = 5
kf = KFold(n_splits=k, shuffle=False)
t = MinMaxScaler()
	
npd = npd.to_numpy()

mse_score = []

myMSE = tf.keras.losses.MeanSquaredError()
 
for train_index , test_index in kf.split(npd):
    
    X_train , X_test = npd[train_index] , npd[test_index]
    
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
     
    hist = vae.fit(X_train, X_train, epochs=EPOCHS, batch_size = BATCH)
    pred_values = vae.predict(X_test)
     
    mse = myMSE(pred_values , X_test)
    mse_score.append(mse)

 
print('\nmse of each fold - {}'.format(mse_score))

mse_score = np.array(mse_score)

print ("\nmean mse: ", np.mean(mse_score))
print ("stddev: ", np.std(mse_score))


print ("fino alla fine")


