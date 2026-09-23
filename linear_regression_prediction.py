import pandas as pd
from sklearn.linear_model import LinearRegression

# Example data
df = pd.DataFrame({
    "year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    "score": [72, 74, 76, 78, 80, 79, 82, 81, 79, 87, 81, 71]
})

# Prepare features and target
X = df[["year"]]
y = df["score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next year
next_year = pd.DataFrame({"year": [2027]})
prediction = model.predict(next_year)[0]

print("Predicted score for 2027:", round(prediction, 2))