import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('titanic.csv')
le = LabelEncoder()
df['Sex'] = le.fit_transform(df.Sex)

x = df[['Pclass','Sex','Age','Fare']]
x = x.fillna(x.mean())      
y = df.Survived
y = y.fillna(y.mean())
# group = df.groupby('Sex')['Survived'].sum()


plt.ylabel('No of people survived')
plt.xlabel('Age')
plt.scatter(x.Fare,y,color='red',marker='o')
# plt.bar(group.index,group.values,color='green')
plt.show()


# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

# model = tree.DecisionTreeClassifier()
# model.fit(x_train,y_train)
# print("Accuracy is ",model.score(x_test,y_test))

