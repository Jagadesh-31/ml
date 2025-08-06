import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv("HR_comma_sep.csv")

# grouped = df.groupby('salary')['left'].sum()

# print(grouped)
# plt.title('Employees Left by Salary Level')
# plt.xlabel('Salary')
# plt.ylabel('Number of Employees Who Left')

# plt.bar(grouped.index, grouped.values, color='orange')

# plt.show()

# group = df.groupby('Department')['left'].sum()

# print(group);

# plt.title('Retention of Employees based on department')
# plt.xlabel('Department')
# plt.ylabel('No of Employees retentioned')
# plt.bar(group.index,group.values,color='yellow')

# plt.show()
X = df.drop('left', axis=1)  
y = df['left']    


X = pd.get_dummies(X)  #only the non-numerical columns gets dummies remaining left unchanged


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = linear_model.LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)


print("Model accuracy:", model.score(x_test, y_test))

