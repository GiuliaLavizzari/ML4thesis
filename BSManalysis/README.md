## [job.sh](https://github.com/GiuliaLavizzari/ML4thesis/blob/7561a4df91d9811d7b0f19b91b7a710a7a3fe6f0/job.sh)  
This script performs the complete BSM analysis: indeed, it computes the loss for the SM events ([lossperbatch.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatch.py)) and for the BSM events ([lossperbathBSM.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/e0327246bc1dca059d2317e9e5687bde6a233e19/lossperbatchBSM.py)). Those losses are used to single out EFT events: these are in fact expected to be recontructed badly by the VAE (thus showing a bigger loss). Therefore, a threshold value is chosen and the events whose loss is greater than this threshold are selected.  

## [finalBSM1.py](https://github.com/GiuliaLavizzari/ML4thesis/blob/7561a4df91d9811d7b0f19b91b7a710a7a3fe6f0/finalBSM1.py).  
Computes the significance sigma as the number of EFT events (LIN+QUAD) divided by the square root of the number of SM events. This is the chosen figure of merit to choose wether a particular VAE model is sensitive to an EFT operetor or not.  
In order to do so, the events have to be correctly normalized:
```python
# example of normalization: SM sample
luminosity = 1000.*350. #luminosity expected in 1/pb
fSM = ROOT.TFile("/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
hSM = fSM.Get("SSWW_SM_nums")
xsecSM = hSM.GetBinContent(1)
sumwSM = hSM.GetBinContent(2)
normSM = 5.* xsecSM * luminosity / (sumwSM) # on test set (0.2*total)
```
It's also possible to compute the minimum value of the wilson coefficient for which the analysis is sensitive to the EFT operator to which the coefficient is associated.
