import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split 

df = pd.read_csv("iris.csv")


x = df.drop('species',axis=1);
# y = pd.get_dummies(df.species);

y = df['species'];

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2);
# 0.95 is underfit
# 0.05 is overfit

model = linear_model.LogisticRegression(max_iter=200)
model.fit(x_train,y_train)


print("Accuracy is : ",model.score(x_test,y_test))

