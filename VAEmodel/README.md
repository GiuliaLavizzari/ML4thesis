# Training the model

# [VAEmodel.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/959a2c89113660b455d04cb86396b2c440d45285/VAEmodel.py)  
The VAE model used. The structure of this model is shown in the following figure:
![Alt Text](https://github.com/GiuliaLavizzari/ML4thesis/blob/5aa6ab696a6b371c9d9f320aad6a5e7f4d0822b8/vaemodel.PNG)
More information on the model can be found at this link: [kerasVAE](https://keras.io/examples/generative/vae/).   

# [training.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/959a2c89113660b455d04cb86396b2c440d45285/training.py)  
Trains the model and saves the encoder and the VAE model, together with a .csv file containing the values of the losses per epoch.  
Dimension of the latent space, number of epochs, batch size and learning rate of the optimizer can be modified here.

Importing the model:
```python
from VAEmodel import * # imports the chosen model
MODEL = 1 # name of the model
```
Saving the data:
```python
# selecting the variables used for the training
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
dfAll = ROOT.RDataFrame("SSWW_SM","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
npy = df.AsNumpy(pd_variables)
npd =pd.DataFrame.from_dict(npy)

# storing the weights of the events
wSM = df.AsNumpy("w")
wpdSM = pd.DataFrame.from_dict(wSM)
```
Splitting the data:
```python
X_train, X_test, y_train, y_test = train_test_split(npd, npd, test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(wpdSM, wpdSM, test_size=0.2, random_state=1)
wx = wx_train["w"].to_numpy()
wxtest = wx_test["w"].to_numpy()
```
Preprocessing:
```python
#logarithm of the kinematic variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1', 'ptl2', 'ptll']:
	npd[vars] = npd[vars].apply(np.log10)

#scaling the data
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)
```
Setting the parameters of the model:
```python
DIM = sys.argv[1] #latent dimension
BATCH = 32 #batch size
EPOCHS = 200 #number of epochs
```
Training the model:
```python
# whole model
vae = VariationalAutoEncoder(original_dim, DIM) # (self, original, latent)
vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=tf.keras.losses.MeanSquaredError())
hist = vae.fit(X_train, X_train, epochs=EPOCHS, batch_size = BATCH)
```
```python
# encoder
encoder = LatentSpace(original_dim, DIM) # (self, original, latent)
z = encoder.predict(X_train)
```
Saving the model:
```python
enc_name = "myenc{}_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(MODEL, DIM, EPOCHS, BATCH)
vae_name = "myvae{}_denselayers_latentdim{}_epoch{}_batchsize{}_log_eventFiltered".format(MODEL, DIM, EPOCHS, BATCH)
csv_name = "myloss{}_training_latentdim{}_epoch{}_batchsize{}_log_eventFiltered.csv".format(MODEL, DIM, EPOCHS, BATCH)
tf.keras.models.save_model(encoder, enc_name) 
tf.keras.models.save_model(vae, vae_name)
np.savetxt(csv_name, hist.history["loss"], delimiter=',')
```

## [trainLosses.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/0ce93379f5cda769863e494969bb69e09575cbd9/VAEmodel/trainLosses.py)
This script allows for saving the values of the MSE (reconstruction loss), the KLD (regularization term) and the total loss (MSE+KLD) computed during the training as a function of the number of epochs. Note that in order to have access to the KLD and MSE values the model has to be set in a different way with respect to the one previously described. The used model is in fact the following:

## [model4Losses.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/7eb1d4ae0b6f06603dd7118ed5753fffa2181d5c/VAEmodel/model4Losses.py)
This model allows for keeping track of the value of the losses during the training. 
