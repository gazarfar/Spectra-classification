# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:17:15 2022

@author: azarf
"""

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot  as plt

def MyPCA(data,label,num_components,plot,description = 'data'):
    mymean = np.mean(data,axis = 1, keepdims = True)
    centered_spectrum = data - mymean
    pca = PCA(n_components = num_components)
    pca.fit(centered_spectrum)
    features = pca.transform(centered_spectrum)   
    percentage = np.sum(pca.explained_variance_ratio_)
    if plot == 1:
        print("Principal component analysis (PCA) is performed to reduce data") 
        print(" and extract features")
        Red = np.zeros((data.shape[0],1))
        Green = np.zeros((data.shape[0],1))
        Blue = np.zeros((data.shape[0],1))
        Red[label == 0] = 0
        Red[label == 1] = 0.8500
        Red[label == 2] = 0.9260
        Green[label == 0] = 0.4470
        Green[label == 1] = 0.3250
        Green[label == 2] = 0.6940
        Blue[label == 0] = 0.7410
        Blue[label == 1] = 0.0980
        Blue[label == 2] = 0.1250
        mycolors = np.append(Red,Green, axis = 1)
        mycolors = np.append(mycolors,Blue, axis = 1)
        plt.figure(description)
        plt.scatter(features[:,0], features[:,1], c = mycolors)
        plt.title('PC analysis of ' + description)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(['product','background','foreign material'])
        plt.show()
    return features, percentage
