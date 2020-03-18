# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:36:45 2020

@author: Baraka
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Book1.csv')
dataset2=pd.read_csv('test.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1:].values
z=dataset2.iloc[:,0:1]

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

y_pred=lin_reg.predict(z)

plt.scatter(X[0:5],z,color='red')
plt.plot(X[0:5],lin_reg.predict(z),color='blue')














