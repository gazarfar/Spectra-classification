"""
Created on Tue Dec 20 13:45:53 2022

@author: azarf
"""

############################################
# Packages & functions & path2data
############################################

import pickle
import numpy as np
import matplotlib.pyplot  as plt
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from data_loader import data_loader
from MyPCA import MyPCA 
from Performance import performance


path2data = 'C:\\P&P optica\\training_data.pkl'

############################################
#Loading the data
############################################
Spectrum, Label, target_names = data_loader(path2data)

############################################
# Data Visualization & Problem Understanding
############################################
print('-------------------------------------------------------------------')
print('Data Visualization')
print('-------------------------------------------------------------------\n')


num_class, num_samples_per_class = np.unique(Label,  return_counts=True)
print('There are ' + str(len(num_class)) + ' classes in the dataset labeled as follows:\n')
print('1) class product labeled as"' + str(num_class[0]) + '" with ' + str(num_samples_per_class[0]) + ' samples')
print('2) class background labeled as"' + str(num_class[1]) + '" with ' + str(num_samples_per_class[1]) + ' samples')
print('3) class foreign material labeled as"' + str(num_class[2]) + '" with ' + str(num_samples_per_class[2]) + ' samples\n')

print('The spectra of each group is as shown below:')
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(0,10):
    indx = rn.randint(0, len(Label))
    if Label[indx] == 0:
        plt.subplot(3,1,1)
        plt.plot(Spectrum[indx,:], color = [0, 0.4470, 0.7410])
        plt.title('product')
    elif Label[indx] == 1:
        plt.subplot(3,1,2)
        plt.plot(Spectrum[indx,:], color = [0.8500, 0.3250, 0.0980])
        plt.title('background')
    else:
        plt.subplot(3,1,3)
        plt.plot(Spectrum[indx,:], color = [0.9290, 0.6940, 0.1250])
        plt.title('foreign material')
        
# Looking at the random samples of the data there are huge variances 
#between the classes of spectra, so principal component analysis 
#reduce the size of input data and extract features, then a simple 
#linear regression neural network can do the classification

############################################
# Creating training and test Sets for
# an initial training and test of the model
############################################

X_train, X_test, y_train, y_test = train_test_split(Spectrum, Label, test_size= 0.33, shuffle=True)


############################################
# Feature extraction by PCA
############################################
print('-------------------------------------------------------------------')
print('Feature Extraction')
print('-------------------------------------------------------------------\n')

num_components = 4 
feature, percentage = MyPCA(Spectrum,Label,num_components,1,'all data')
feature_train, percentage_train = MyPCA(X_train,y_train,num_components,0,'training')
feature_test, percentage_test = MyPCA(X_test,y_test,num_components, 0,'test')

print('The first ' + str(num_components) +'PC components includes ' + str(percentage_train) + '% of the variations in the spectra')
print('As shown in the PCA figure by projecting the spectra to PC coordinate the spectra are already grouped into three clusters')
print('Since the clusters can be seperated by straight lines, a Logistic Regression model with cross validation will work perfectly for classification')
print('-------------------------------------------------------------------')

# ############################################
# # Logistic Regression initial model training
# ############################################ 
print('-------------------------------------------------------------------')
print('Training the model by 66% of the data and testing on the rest')
print('-------------------------------------------------------------------\n')

model = LogisticRegressionCV(cv=5, random_state=0).fit(feature_train, y_train.flatten());


conf_max = performance(model,feature_test, y_test, target_names)

#saving the model
filename = 'model'
#pickle.dump(model, open(filename, 'wb'))

