# BSM analysis
## [job.sh](https://github.com/GiuliaLavizzari/ML4thesis/blob/7561a4df91d9811d7b0f19b91b7a710a7a3fe6f0/job.sh)  
This script performs the complete BSM analysis: indeed, it computes the losses and weights for the SM events ([lossperbatch.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatch.py)) and for the BSM events ([lossperbathBSM.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatchBSM.py)). Those values of the loss are the ones used to single out EFT events: a threshold value is chosen and the events whose loss is greater than this threshold are selected.  

## [lossperbatch.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/32ea0867d4b0001347ba41e03769d39a5203e16c/BSManalysis/lossperbatch.py)
This script retrieves the value of the total loss for each event:
```python
class LossPerBatch(tf.keras.callbacks.Callback):
    def on_test_batch_end(self, batch, logs=None):
        self.eval_loss.append(logs["loss"])
```
```python
mylosses = LossPerBatch()
vae.evaluate(X_test,X_test,batch_size=1,callbacks=[mylosses],verbose=0) #by setting the batch_size=1 it's possible to retrieve the loss for each event
```
Subsequently, the losses and weights are saved. The saved EFT weights are yet to be multiplied by cW or cW^2.  
The same happens for [lossperbathBSM.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatchBSM.py).


## [finalBSM1.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/7561a4df91d9811d7b0f19b91b7a710a7a3fe6f0/finalBSM1.py).  
This script employs the losses and weights computed and saved with [lossperbath.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatch.py) and [lossperbathBSM.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatchBSM.py) and plots the loss function, computes the significance sigma and the minimum value of the wilson coefficient for which the VAE model is sensitive to the EFT operator.  

First, the events need to be correctly normalized. The correct normalization of the events is given by the following factor, that is to be multiplied to the weights of the events (together with the cW and cW^2 coefficients in the cases of the EFT events)
```python
# example of normalization: SM sample
luminosity = 1000.*350. #luminosity expected in 1/pb
fSM = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
hSM = fSM.Get("SSWW_SM_nums")
xsecSM = hSM.GetBinContent(1)
sumwSM = hSM.GetBinContent(2)
normSM = 5.* xsecSM * luminosity / (sumwSM) # on test set (0.2*total)
```

The significance sigma is computed as the number of EFT (LIN + QUAD) events whose loss is bigger than a selected threshold, divided by the square root of the number of SM events above the same threshold. This is the chosen figure of merit to compare the sensitivity of different VAE models to a particular operator.  
The error on sigma due to the fluctuations of the number of events in the Monte Carlo samples is also computed.  

It's also possible to compute the minimum value of the wilson coefficient for which the analysis is sensitive to the EFT operator (where sensitivity is defined as having at least one value of sigma>3).
