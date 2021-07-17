import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import ROOT
import sys
import numpy as np
import pandas as pd

#taking the model
from newVAE2 import *
MODEL = 2

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

nEntries = 100000000000000
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
#t = StandardScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

#VAE params:
DIM = sys.argv[1] #latent dimension
BATCH = 32 #batch size
EPOCHS = 200 #number of epochs
#EPOCHS = int(EPOCHS)

n_inputs = npd.shape[1] 
original_dim = n_inputs

vae = VariationalAutoEncoder(original_dim, DIM) # (self, original, latent)
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=tf.keras.losses.MeanSquaredError())
hist = vae.fit(X_train, X_train, epochs=EPOCHS, batch_size = BATCH)

encoder = LatentSpace(original_dim, DIM) # (self, original, latent)
z = encoder.predict(X_train)

enc_name = "myenc{}_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(MODEL, DIM, EPOCHS, BATCH)
vae_name = "myvae{}_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(MODEL, DIM, EPOCHS, BATCH)
csv_name = "myloss{}_training_latentdim{}_epoch{}_batchsize{}_log_eventFiltered.csv".format(MODEL, DIM, EPOCHS, BATCH)
#vae_name = "BEST_STDsc_vae_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(DIM, EPOCHS, BATCH)
#csv_name = "BEST_STDsc_loss_training_latentdim{}_epoch{}_batchsize{}_log_eventFiltered.csv".format(DIM, EPOCHS, BATCH)

tf.keras.models.save_model(encoder, enc_name) 
tf.keras.models.save_model(vae, vae_name)
np.savetxt(csv_name, hist.history["loss"], delimiter=',')

print ("fino alla fine")


