import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#loading data

df = pd.read_csv("homeprices.csv")

print(df)

#preprocessing data 

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

print(df)

#model

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df['price'])#df.drop('price',axis='columns')==df[['area','bedrooms','age']]

print(reg.coef_,reg.intercept_,'predicted value is ',reg.predict([[3000,3,5]]))