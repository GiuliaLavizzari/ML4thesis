<font size="6"> **STILL WORKING ON IT** </font> 
  
    
http://github.com - automatic!
[caccola](http://github.com)

```python
print ("hello world")
for i in len(ciaone)
    ciaone = ciaone + 1
```

Quick overview of the scripts in this repo. More detailed information is given in the additional COMMENTS/COMMITS of each of them.

[VAEmodel.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/959a2c89113660b455d04cb86396b2c440d45285/VAEmodel.py)  
The VAE model used. The structure of this model is shown in the following figure:
![Alt Text](https://github.com/GiuliaLavizzari/ML4thesis/blob/5aa6ab696a6b371c9d9f320aad6a5e7f4d0822b8/vaemodel.PNG)
More information on the model can be found at this link: [kerasVAE](https://keras.io/examples/generative/vae/).   


[training.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/959a2c89113660b455d04cb86396b2c440d45285/training.py)
Trains the model and saves the encoder and the VAE model, together with a .csv file containing the values of the losses per epoch.  
More specifically:  
Importing the model:
```python
from VAEmodel import * # imports the chosen model
MODEL = 1 # name of the model
```
```python
# storing the input data by choosing the variables of interest among the ones stored in the ntuple
pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']#,'phij1', 'phij2', 'w']
dfAll = ROOT.RDataFrame("SSWW_SM","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
npy = df.AsNumpy(pd_variables)
# storing the weights of the events
wSM = df.AsNumpy("w")
```


tr1.py
https://github.com/GiuliaLavizzari/ML4thesis/blob/6b18dfc6d8ba09b1288ed3d83d6402d7b31c76fa/tr1.py
training
The saved model is characterized by two (main) numbers: 
modelN = defining what version of the model I’m testing
DIM = dimension of the latent space (taken as an input: DIM = argv[1]).
These two numbers often serve as inputs of the various scripts used for the analysis

plotCorrelationMatrix.py
https://github.com/GiuliaLavizzari/ML4thesis/blob/6b18dfc6d8ba09b1288ed3d83d6402d7b31c76fa/plotCorrelationMatrix.py
plots correlation matrix

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

