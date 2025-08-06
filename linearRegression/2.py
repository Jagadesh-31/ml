import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("canada_per_capita_income.csv")

plt.xlabel("year")
plt.ylabel("income")
plt.scatter(df.year, df.income, color='red', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['income'])

m = reg.coef_
c = reg.intercept_


x_vals = np.linspace(1970, 2020, 100)
y_vals = m * x_vals + c
plt.plot(x_vals, y_vals, color='blue')


predicted_income_2000 = reg.predict([[2000]])
print(f"Predicted income in 2000: {predicted_income_2000[0]}")


plt.scatter(2000, predicted_income_2000[0], color='green', marker='o', s=100, label='Prediction (2000)')
plt.legend()

plt.show()

