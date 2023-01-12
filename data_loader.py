# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:40:57 2022

@author: azarf
"""
import pickle

def data_loader(path):
    target_names = ['product','background','foreign material']

    with open(path, 'rb') as reader:
        data = pickle.load(reader)

    Spectrum = data['X']
    Label = data['Y']
    Label[Label == 5]=2

    return Spectrum, Label, target_names