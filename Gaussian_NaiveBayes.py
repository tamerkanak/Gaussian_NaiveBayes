# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:54:20 2023

@author: tamer
"""

import numpy as np

X = np.array([
              [0, 0, 0, 0],
              [0, 0, 0, 1],
              [1, 0, 0, 0],
              [2, 1, 0, 0],
              [2, 2, 1, 0],
              [2, 2, 1, 1],
              [1, 2, 1, 1],
              [0, 1, 0, 0],
              [0, 2, 1, 0],
              [2, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 1, 0, 1],
              [1, 0, 1, 0],
              [2, 1, 0, 1]
])

Y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X,Y)

print(gnb.predict([[0,0,1,0]]))
