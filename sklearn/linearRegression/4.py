import pandas as pd
import numpy as np
from word2number import w2n
from sklearn import linear_model


df = pd.read_csv("hiring.csv")


df.experience = df.experience.fillna('zero')
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())


df.experience = df.experience.apply(w2n.word_to_num)
#converts word to number until we do df.to_csv("filename") same file exits in the repo


reg = linear_model.LinearRegression()
reg.fit(df.drop('salary', axis=1), df['salary'])


predicted = reg.predict([[2, 9, 6]])
print("Predicted Salary:", predicted[0])
