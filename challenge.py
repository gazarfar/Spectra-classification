

import pickle
from sklearn.decomposition import PCA
import numpy as np
import sklearn.linear_model 



class SpecPredict(object):
    """ A coordinate made up of an x and y value """
    def __init__(self,path):
        """ Sets the x and y values """
        self.path = path
        self.model = pickle.load(open(path, 'rb'))
    def __str__(self):
        """ Returns a string representation of self """
        return "linear logistic regression for spectra classification"
    def predict(self, testing_data):
        """ Returns the euclidean distance between two points """
        mymean = np.mean(testing_data,axis = 1, keepdims = True)
        centered_spectrum = testing_data - mymean
        pca = PCA(n_components = 4)
        pca.fit(testing_data)
        features = pca.transform(centered_spectrum)
        return self.model.predict(features)
    def predict_proba(self, testing_data):
        """ Returns the euclidean distance between two points """
        mymean = np.mean(testing_data,axis = 1, keepdims = True)
        centered_spectrum = testing_data - mymean
        pca = PCA(n_components = 4)
        pca.fit(testing_data)
        features = pca.transform(centered_spectrum)
        return self.model.predict_proba(features)

