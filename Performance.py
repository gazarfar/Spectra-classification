# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:29:53 2022

@author: azarf
"""
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from itertools import cycle



def performance(model,X_test, y_test, target_names):
    ########################################
    #Confusion matrix
    ########################################
    y = model.predict(X_test)
    conf_mat = confusion_matrix(y_test.ravel(),y)
    conf_mat = conf_mat/np.sum(conf_mat, axis = 1)*100
    print('The confusion matrix of test set is:')
    print(conf_mat)
    
    ########################################
    #ROC curve & AUC
    ########################################
    
    y_score = model.predict_proba(X_test)
    label_binarizer = LabelBinarizer().fit(y_test.ravel())
    y_onehot_test = label_binarizer.transform(y_test.ravel())

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for class_id, color in zip(range(3), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
         )
   
    
    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()
   
    return conf_mat

# print(conf_mat)
