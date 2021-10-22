
**still working on it**  
*still working on it*
  
    
http://github.com - automatic!
[caccola](http://github.com)

```python
print ("hello world")
for i in len(ciaone)
    ciaone = ciaone + 1
```

Quick overview of the scripts in this repo. More detailed information is given in the additional COMMENTS/COMMITS of each of them.
# VAE model
[VAEmodel.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/959a2c89113660b455d04cb86396b2c440d45285/VAEmodel.py)  
The VAE model used. The structure of this model is shown in the following figure:
![Alt Text](https://github.com/GiuliaLavizzari/ML4thesis/blob/5aa6ab696a6b371c9d9f320aad6a5e7f4d0822b8/vaemodel.PNG)
More information on the model can be found at this link: [kerasVAE](https://keras.io/examples/generative/vae/).   

# training the model
[training.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/959a2c89113660b455d04cb86396b2c440d45285/training.py)
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

# Useful plots

## correlation matrix
[plotCorrelationMatrix.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/37b776ca48e7d9a03df717210364f3f3f63dffee/plotCorrelationMatrix.py)


myPlots.py
https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/myPlots.py
Plots in/out distributions, differences between input and output, latent vars (similar to the aforementioned plotCombinedSamples)

plotROC.py
https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/plotROC.py
Plots SM efficiency vs BSM efficiency, just as the aforementioned lossCutROC. In addition, it computes the AUC value.


job.sh
https://github.com/GiuliaLavizzari/ML4thesis/blob/3a61990021a893f2ebd88d5bdae61e7320a9e43e/job.sh
Complete BSM analysis: calls lossperbatch.py, lossperbatchBSM.py, finalBSM1.py

lossperbatch.py
https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatch.py
Computes and saves as .csv the loss and the weights for each event of the SM sample
(input: modelN = number of the model that needs to be tested, dim = dimension of the latent space of the chosen model)
(output: lossSM, weightsSM)

lossperbatchBSM.py
https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatchBSM.py
Computes and saves as .csv the loss and the weights for each event of the LIN and QUAD samples, separately
(input: modelN, dim, operator)
(output: lossLIN, weightsLIN, lossQUAD, weightsQUAD)

finalBSM1.py
https://github.com/GiuliaLavizzari/ML4thesis/blob/e13f1bb5fd3a2f041d40d61fa3648e21d9e7d28a/finalBSM1.py
→ computes sigma and its error
→ computes the correct event normalization (172-200)
→ plots the loss function (210-252)
→ computes the minimum value of c_{op} for which the analysis is sensitive to the EFT operator (314-326)
→ plots sigma(k) (277-311) and sigmaMAX(cW) (332-435)
(input: modelN, dim, operator, lossSM, lossLIN, lossQUAD, weightsSM, weightsLIN, weightsQUAD)



For a quick use:
→ Train the model with tr1.py
→ Use myPlots.py to see the input/output distributions and the latent space. It also computes the loss function for each event
→ run job1.sh to get the full BSM analysis

