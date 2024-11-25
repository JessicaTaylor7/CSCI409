import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('DataSet.csv')

# Drop rows with missing values
data = data.dropna(subset=['Abstract', 'Article Citation Count'])

# Log-transform citation counts to handle skewness
data['Citation_Log'] = np.log1p(data['Article Citation Count'])

# TF-IDF Vectorization
# The TF-IDF vectorizer is used to convert text data into numerical form, where each word is represented as a feature.
# max_features=5000 limits the number of features (terms) used to the top 5000 most important words (based on TF-IDF score).
# stop_words='english' removes common English stop words (e.g., 'the', 'is', 'in') from the analysis.
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(data['Abstract']).toarray()  # Transform the 'abstract' column into a matrix of features.
y = data['Citation_Log']  # The target variable (citation count after log transformation).

# Train-Test Split
# We split the data into training and testing sets, with 80% for training and 20% for testing.
# The random_state ensures reproducibility by setting a fixed seed for random operations.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- RANDOM FOREST MODEL ----------
# Create and train the Random Forest model.
# n_estimators=100 means the model will use 100 decision trees in the ensemble.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Fit the model using the training data.

# Predict with Random Forest
y_pred_rf = rf_model.predict(X_test)  # Use the trained model to predict the citation count for the test data.

# ---------- LINEAR REGRESSION MODEL ----------
# Create and train the Linear Regression model.
# Linear regression is a simpler model that assumes a linear relationship between features and target variable.
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # Fit the model using the training data.

# Predict with Linear Regression
y_pred_lr = lr_model.predict(X_test)  # Use the trained model to predict the citation count for the test data.

# ---------- PERFORMANCE COMPARISON ----------
# Reverse log transformation (optional, for interpretability)
# The log transformation was applied to the citation count earlier. We now reverse this transformation to bring it back to the original scale.
y_test_exp = np.expm1(y_test)  # Reverse the log transformation on the true values.
y_pred_rf_exp = np.expm1(y_pred_rf)  # Reverse the log transformation on Random Forest predictions.
y_pred_lr_exp = np.expm1(y_pred_lr)  # Reverse the log transformation on Linear Regression predictions.

# Metrics for Random Forest
# Evaluate model performance using Root Mean Squared Error (RMSE) and R² score.
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))  # RMSE gives an idea of the error magnitude.
r2_rf = r2_score(y_test, y_pred_rf)  # R² score indicates how well the model explains the variance of the target.

# Metrics for Linear Regression
# Same metrics applied to the Linear Regression model.
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))  # RMSE for Linear Regression model.
r2_lr = r2_score(y_test, y_pred_lr)  # R² score for Linear Regression model.

# Print Results
# Display the performance metrics for both models.
print("Random Forest Results:")
print(f"  RMSE: {rmse_rf}")  # Print RMSE for Random Forest.
print(f"  R² Score: {r2_rf}\n")  # Print R² score for Random Forest.

print("Linear Regression Results:")
print(f"  RMSE: {rmse_lr}")  # Print RMSE for Linear Regression.
print(f"  R² Score: {r2_lr}")  # Print R² score for Linear Regression.

# ---------- VISUALIZATION ----------
# Plot True vs Predicted for Random Forest
# This visualizes the relationship between the actual and predicted values for Random Forest.
# It helps to see how closely the model's predictions match the true values.
plt.figure(figsize=(12, 6))

# Subplot 1: Random Forest performance plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5)  # Scatter plot showing the true vs predicted values.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line representing perfect prediction.
plt.xlabel("True Citation Log")  # Label for x-axis (True values).
plt.ylabel("Predicted Citation Log")  # Label for y-axis (Predicted values).
plt.title("Random Forest Performance")  # Title for the plot.

# Subplot 2: Linear Regression performance plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lr, alpha=0.5)  # Scatter plot for Linear Regression predictions.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line for comparison.
plt.xlabel("True Citation Log")  # Label for x-axis (True values).
plt.ylabel("Predicted Citation Log")  # Label for y-axis (Predicted values).
plt.title("Linear Regression Performance")  # Title for the plot.

# Adjust layout to ensure the plots don't overlap
plt.tight_layout()
plt.show()  # Display the plots.
