
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#import sys
import numpy as np
import pandas as pd

#from matplotlib import pyplot as plt

#from VAE_new_model import *
import ROOT


def sort_by_absolute(df, index):
    df_abs = df.apply(lambda x: abs(x))
    df_abs = df_abs.sort_values('reconstruction_loss', ascending = False)
    df = df.loc[df_abs.index,:]
    return df


cW = 0.5 #0.3
nEntries = 2000

pd_variables = ['deltaetajj', 'deltaphijj', 'etaj1', 'etaj2', 'etal1', 'etal2',
       'met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll',"w"]#,'phij1', 'phij2']

dfAll = ROOT.RDataFrame("SSWW_SM","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_SM.root")
df = dfAll.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_QUAD = ROOT.RDataFrame("SSWW_cW_QU","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_cW_QU.root")
dfBSM_QUAD = dfBSMAll_QUAD.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")
dfBSMAll_LIN = ROOT.RDataFrame("SSWW_cW_LI","/gwpool/users/glavizzari/Downloads/ntuple_SSWW_cW_LI.root")
dfBSM_LIN = dfBSMAll_LIN.Filter("ptj1 > 30 && ptj2 >30 && deltaetajj>2 && mjj>200")

SM =pd.DataFrame.from_dict(df.AsNumpy(pd_variables))
BSM_quad=pd.DataFrame.from_dict(dfBSM_QUAD.AsNumpy(pd_variables))
BSM_lin=pd.DataFrame.from_dict(dfBSM_LIN.AsNumpy(pd_variables))
SM = SM.head(nEntries)
BSM_lin = BSM_lin.head(nEntries)
BSM_quad = BSM_quad.head(nEntries)
All_BSM = pd.concat([BSM_quad, BSM_lin], keys=['Q','L'])



#using logarithm of some variables
for vars in ['met', 'mjj', 'mll',  'ptj1', 'ptj2', 'ptl1',
       'ptl2', 'ptll']:
    All_BSM[vars] = All_BSM[vars].apply(np.log10)
    SM[vars] = SM[vars].apply(np.log10)

#Rescaling weights for the Wilson coefficient
All_BSM["w"].loc["L"] = All_BSM["w"].loc["L"].to_numpy()*cW
All_BSM["w"].loc["Q"] = All_BSM["w"].loc["Q"].to_numpy()*cW*cW

weights = All_BSM["w"].to_numpy()
weights_SM = SM["w"].to_numpy()

#concatenating the SM + BSM part
All = pd.concat([SM,All_BSM])


weights_all = All["w"].to_numpy()
SM.drop('w',axis='columns', inplace=True)
All_BSM.drop('w',axis='columns', inplace=True)
All.drop('w',axis='columns', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(SM,SM,test_size=0.2, random_state=1)
wx_train, wx_test, wy_train, wy_test = train_test_split(weights_SM, weights_SM, test_size=0.2, random_state=1)
#All_BSM = All_BSM.to_numpy()
_, All_BSM_test,_,_  = train_test_split(All_BSM,All_BSM,test_size=0.2, random_state=1)
w_train, w_test,_,_ = train_test_split(weights, weights,test_size=0.2, random_state=1)
All_test = np.concatenate((All_BSM_test,X_test))
weight_test = np.concatenate((w_test, wx_test))

t = MinMaxScaler()
t.fit(X_train)
#X_test=t.transform(X_test)
All_test = t.transform(All_test)
All_BSM_test = t.transform(All_BSM_test)

model = tf.keras.models.load_model('finalvae1_denselayers_latentdim7_epoch250_batchsize32_log_eventFiltered')


# scelgo l'evento che contribuisce maggiormente alla loss
# in questo evento scelgo le 5 variabili che contribuiscono maggiormente al valore della loss

#selecting the event that has the greatest lf value
output = model.predict(All_BSM_test)
rec_err = np.linalg.norm(All_BSM_test - output, axis = 1)
idx = list(rec_err).index(max(rec_err)) # gets the index of the max val of rec_err

#sorting the variables according to the size of their contribution to the lf
df = pd.DataFrame(data = (All_BSM_test[idx]-output[idx]), index = All.columns, columns = ['reconstruction_loss'])
top_5_features = sort_by_absolute(df, idx).iloc[:5,:]
print ("\n\ntop 5 features in the worst reconstructed event: ")
print(top_5_features.T)


#class shap.KernelExplainer(model, data, link=<shap.utils._legacy.IdentityLink object>, **kwargs)
#Uses the Kernel SHAP method to explain the output of any function.
#Kernel SHAP is a method that uses a special weighted linear regression to compute the importance of each feature. The computed importance values are Shapley values from game theory and also coefficents from a local linear regression.

#shap_values(X, **kwargs)
#Estimate the SHAP values for a set of samples. For models with a single output this returns a matrix of SHAP values (# samples x # features). Each row sums to the difference between the model output for that sample and the expected value of the model output (which is stored as expected_value attribute of the explainer). For models with vector outputs this returns a list of such matrices, one for each output.

#SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature.

import shap
data_summary = shap.kmeans(All_BSM_test, 100) #clustera All_BSM_test in 100 clusters
shaptop5features = pd.DataFrame(data = None)
 
for i in top_5_features.index: # i e' indice tipo ptjj, etajj...
    weights = model.get_layer('el1').get_weights()
    ## make sure the weight for the specific one input feature is set to 0
    feature_index = list(df.index).index(i)
    print(feature_index, i)
    updated_weights = weights[:][0]
    updated_weights[feature_index] = [0]*len(updated_weights[feature_index])
    model.get_layer('el1').set_weights([updated_weights, weights[:][1]])
    
    ## determine the SHAP values
    explainer_autoencoder = shap.KernelExplainer(model.predict, data_summary)
    #print(All_BSM_test[idx,:])
    shap_values = explainer_autoencoder.shap_values(All_BSM_test[idx,:])
    print (shap_values)

    ## build up pandas dataframe
    shaptop5features[str(i)] = pd.Series(shap_values[feature_index]) #column
    
shaptop5features.index = df.index
print(shaptop5features)
## Ditch the confusing way of splitting into contributing and offsetting groups in the original paper
print (top_5_features)
top_5_features['contributing'] = (top_5_features['reconstruction_loss'] > 0).astype(int)
print (top_5_features)
## Any postive value is contributing and any negative value is offsetting
for i in range(5):
    if top_5_features['contributing'][i] == 0:
        shaptop5features[top_5_features.index[i]] = shaptop5features[top_5_features.index[i]] * (-1)

print(shaptop5features)
shap.summary_plot(shaptop5features.T.values, shaptop5features.index, max_display = 10, plot_type = "bar")
