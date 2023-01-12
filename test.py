# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:06:29 2022

@author: azarf
"""

from challenge import SpecPredict
import matplotlib.pyplot  as plt
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from data_loader import data_loader
from MyPCA import MyPCA 
from Performance import performance



path2model = 'C:\\P&P optica\\model'
path2data = 'C:\\P&P optica\\training_data.pkl'

testing_data, testing_Label, target_names = data_loader(path2data)


model = SpecPredict(path2model)
predicted_label = model.predict(testing_data)



feature_test, percentage_test = MyPCA(testing_data,testing_Label, 4, 0,'test')
conf_max = performance(model,feature_test, testing_Label, target_names)

# testing_data will be a [sample x feature] numpy array and should
# return a [sample x class] numpy array.