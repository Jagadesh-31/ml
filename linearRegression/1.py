import pandas as pd
import numpy as np
from sklearn import linear_model  
import matplotlib.pyplot as plt  


df = pd.read_csv("data.csv")


reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])


m = reg.coef_[0]
c = reg.intercept_


plt.xlabel("Area (in sqft)")
plt.ylabel("Price (in $)")
plt.scatter(df.area, df.price, color='red', marker='+')


x_vals = np.linspace(0, 5000, 100)  #creates evenly spaced 100 points array inclusive of 0 and 5000
y_vals = m * x_vals + c;


areas_data = pd.read_csv("prediction.csv")
areas_data["price"] = reg.predict(areas_data)

areas_data.to_csv('prediction.csv')

plt.scatter(areas_data.area,areas_data.price,color='green',marker='o')

plt.plot(x_vals, y_vals, color='blue')

plt.show()



area_df.to_csv("prediction.csv")