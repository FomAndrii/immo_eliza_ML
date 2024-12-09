import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load the dataset
df = pd.read_csv("./data/immoweb_data_processed.csv")

# Display correlation matrix
corr = df.corr()
ax = sns.heatmap(
    corr,
    vmin=-1,
    vmax=1,
    center=0,
    cmap=sns.diverging_palette(40, 220, n=200),
    square=True,
)
print(corr)

# Heatmap with annotations
plt.subplots(figsize=(16, 8))
sns.heatmap(
    df.corr(),
    annot=True,
    linewidths=0.05,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
)
plt.show()

# Select features (X) and target (y)
X = df[["Type_of_Property", "Living_Area", "Region_Code"]]
y = df["Price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.22, random_state=42
)

# Create and train the Decision Tree model
tr_regressor = DecisionTreeRegressor(random_state=42)
tr_regressor.fit(X_train, y_train)

# Predict using the trained model
pred_train = tr_regressor.predict(X_train)
pred_test = tr_regressor.predict(X_test)

# Evaluate the model on the training set
r2_train = r2_score(y_train, pred_train)
mse_train = mean_squared_error(y_train, pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, pred_train)

# Evaluate the model on the testing set
r2_test = r2_score(y_test, pred_test)
mse_test = mean_squared_error(y_test, pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, pred_test)

# Display the results
print("Training Set Performance:")
print("R² Score:", r2_train)
print("Mean Squared Error (MSE):", mse_train)
print("Root Mean Squared Error (RMSE):", rmse_train)
print("Mean Absolute Error (MAE):", mae_train)

print("\nTesting Set Performance:")
print("R² Score:", r2_test)
print("Mean Squared Error (MSE):", mse_test)
print("Root Mean Squared Error (RMSE):", rmse_test)
print("Mean Absolute Error (MAE):", mae_test)

# Plot actual vs predicted prices for the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_test, color="orange", alpha=0.7, label="Predicted vs Actual")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "g--",
    lw=2,
    label="Ideal Prediction",
)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Decision Tree: Actual vs Predicted Prices (Test Set)")
plt.legend()
plt.show()

# Plot actual vs predicted prices for the train set
plt.figure(figsize=(8, 6))
plt.scatter(y_train, pred_train, color="blue", alpha=0.7, label="Predicted vs Actual")
plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    "g--",
    lw=2,
    label="Ideal Prediction",
)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Decision Tree: Actual vs Predicted Prices (Train Set)")
plt.legend()
plt.show()