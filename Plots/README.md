# Plots
## [plotCorrelationMatrix.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/37b776ca48e7d9a03df717210364f3f3f63dffee/plotCorrelationMatrix.py)  
Plots the correlation matrix considering all the samples (SM, LIN, QUAD).  
A column labelled "sample" has been added to the dataframe containing the data: the value of the "sample" variable is set as 0 for the events belonging to the SM sample, 1 for LIN and 2 for QUAD. This allows for checking the correlation between each variable and the three samples.
Adding the "Sample" column:
```python
BSM_quad["Sample"] = 2 #adds a column ('sample') with value 2
BSM_lin["Sample"] = 1
SM["Sample"] = 0
```
Plotting the matrix:
```python
#concatenating dataframes and plotting correlation matrix with seaborn
All = pd.concat([SM,BSM_quad,BSM_lin], keys=["S","Q","L"])
All_corrM = All.corr()
import seaborn as sn
sn.heatmap(All_corrM, annot=True)
```

## [myPlots.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/main/myPlots.py)  
Plots input vs output distributions, the difference between input and output values (both for SM and BSM distributions), and the latent variables.  
The data belonging to the SM sample are split and scaled just as before the training. The splitting must be the same of the training in order to run the trained model on different data than those used for the training. For the sake of consistency, the same scaling factor computed during the training has to be applied to all the samples on which the model is tested.
```python
t = MinMaxScaler()
t.fit(X_train) # fit on the training dataset
# this very same scaling factor is then applied to all the datasets
X_train = t.transform(X_train) 
X_test = t.transform(X_test)
All_test = t.transform(All_test) # contains X_test, LIN and QUAD
```

In order to plot the output variables the trained model has to be loaded and used to predict the outputs:
```python
vae = tf.keras.models.load_model(model_name) # loading the VAE model
out = vae.predict(All_test) # all data
out_SM = vae.predict(X_test) # only SM (test set)
```
The same happens for the latent variables:
```python
enc = tf.keras.models.load_model(encname) # loading the encoder
latentALL = enc.predict(All_test)
```


## [plotROC.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/5bb127d8f3484f1ad6b16a71fe44c395a57a308d/plotROC.py)  
Computes the SM and BSM efficiency as follows:
* effSM = sumWSM*/sumWSM  
where:  
sumWSM = sum of the weights of all the SM events  
sumWSM* = sum of the weights of the SM events whose loss exceeds a chosen threshold value (cut)  

* effBSM = sumWBSM*/sumWBSM  
where:  
sumWBSM = sum of the weights of all the BSM=SM+EFT events  
sumWBSM* = sum of the weights of the BSM=SM+EFT events whose loss exceeds a chosen threshold value (cut)  

The roc curve is given by the plot of the BSM efficiency versus SM efficiency; the AUC value is also computed.
