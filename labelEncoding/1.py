import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


df = pd.read_csv("carprices.csv")
print("Original Data:\n", df)

# Step 2: Define X (features) and y (target)
X = df[['Car Model', 'Mileage', 'Age']] # Notice 'Car Model' is still a string here
y = df['Price']

# Step 3: Apply OneHotEncoder directly on the 'Car Model' string column (index 0)
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['Car Model'])  # Directly encode the string column
    ],
    remainder='passthrough'  # Keep Mileage and Age
)

X_transformed = ct.fit_transform(X)
print("\nAfter OneHotEncoding:\n", X_transformed)


model = LinearRegression()
model.fit(X_transformed, y)


print("\nAccuracy: ", model.score(X_transformed,y)*100)
