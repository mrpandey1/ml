#importing libraries
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

#importing datasets
dataset=pd.read_csv('C:\\Users\\Baraka\\Desktop\\Machine Learning A-Z Template Folder\\Part 1 - Data Preprocessing\\Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1:].values

#spliting train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

"""
for missing values
from sklearn.preprocessing import Imputer 
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,1:])
x[:,1:]=imputer.transform(x[:,1:])
"""
"""
#for encoding the string values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder=LabelEncoder()
x[:,0]=label_encoder.fit_transform(x[:,0])
y[:,0]=label_encoder.fit_transform(y[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
onehotencoder=OneHotEncoder(categorical_features=[-1])
y=onehotencoder.fit_transform(y).toarray()
"""
"""
#for scaling large values between -1 to +1
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

"""













