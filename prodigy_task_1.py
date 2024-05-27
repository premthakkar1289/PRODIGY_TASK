import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
df = pd.read_csv("After_Modifications.csv")
df.head(10)

y = df['price']
X = df.drop(columns = ['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

linear = LinearRegression()

linear.fit(X_train, y_train)

y_pred = linear.predict(X_test)

r_squared = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residual Plot")
plt.show()

# Print evaluation metrics
print("R-squared:", r_squared)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

num_features = 5
rfe = RFE(linear, n_features_to_select = num_features)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_]
selected_features

y = df['price']
X_selected = X[selected_features]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_selected, y, test_size = 0.2, random_state = 42)
linear2 = LinearRegression()
# Fit the model
linear2.fit(X_train2, y_train2)
# Make predictions
# Make predictions on the test set
y_pred2 = linear2.predict(X_test2)

# Calculate R-squared
r_squared2 = r2_score(y_test2, y_pred2)

# Calculate mean absolute error
mae2 = mean_absolute_error(y_test2, y_pred2)

# Calculate mean squared error
mse2 = mean_squared_error(y_test2, y_pred2)

# Calculate root mean squared error
rmse2 = np.sqrt(mse2)

# Plot predicted vs. actual values
plt.scatter(y_test2, y_pred2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()

# Create a residual plot
residuals2 = y_test2 - y_pred2
plt.scatter(y_pred2, residuals2)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residual Plot")
plt.show()

# Print evaluation metrics
print("R-squared:", r_squared2)
print("Mean Absolute Error:", mae2)
print("Mean Squared Error:", mse2)
print("Root Mean Squared Error:", rmse2)

desired_alpha = 0.1
model = Lasso(alpha = desired_alpha)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print the coefficients of the selected features (non-zero coefficients)
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
selected_features = coefficients[coefficients['Coefficient'] != 0]
print("Selected Features:")
print(selected_features)

# The correlation matrix
correlation_matrix = df.corr()

# Calculate the correlation with the target variable 'price'
target_correlation = correlation_matrix['price']

# Select features with an absolute correlation greater than or equal to 0.5 (both positive and negative)
relevant_features = target_correlation[(target_correlation >= 0.5) | (target_correlation <= -0.5)].index.tolist()
relevant_features

plt.figure(figsize = (12, 10))

# Heatmap
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm")

# Title
plt.title("Correlation Matrix")

# Show
plt.show()

y = df['price']
X_selected2 = X[['sqft_living', 'grade', 'sqft_above', 'sqft_living15']]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X_selected2, y, test_size = 0.2, random_state = 42)
# Initializing and training a linear regression model
linear3 = LinearRegression()
# Fit the model
linear3.fit(X_train3, y_train3)
# Make predictions
# Make predictions on the test set
y_pred3 = linear3.predict(X_test3)

# Calculate R-squared
r_squared3 = r2_score(y_test3, y_pred3)

# Calculate mean absolute error
mae3 = mean_absolute_error(y_test3, y_pred3)

# Calculate mean squared error
mse3 = mean_squared_error(y_test3, y_pred3)

# Calculate root mean squared error
rmse3 = np.sqrt(mse3)

# Plot predicted vs. actual values
plt.scatter(y_test3, y_pred3)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()

# Create a residual plot
residuals3 = y_test3 - y_pred3
plt.scatter(y_pred3, residuals3)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residual Plot")
plt.show()

# Print evaluation metrics
print("R-squared:", r_squared3)
print("Mean Absolute Error:", mae3)
print("Mean Squared Error:", mse3)
print("Root Mean Squared Error:", rmse3)

k = KFold(n_splits = 10, shuffle = True, random_state = 42)

# Use cross_val_score to measure the accuracy of the model.
accuracy = cross_val_score(linear, X, y, cv = k) # Splitting is taken care of

# Print the average accuracy score  
print("The accuracy using cross validation is: ", accuracy)

# The correct method to print the cross validation accuracy
average_accuracy = np.mean(accuracy)
print(f'Average Accuracy: {average_accuracy:.2f}')
train_sizes, train_scores, test_scores = learning_curve(linear, X, y, cv = k, train_sizes = np.linspace(0.1, 1.0, 10)) # Automatically handles splitting the data.

# Average score of training and testing
avg_train = np.mean(train_scores, axis = 1)
avg_test = np.mean(test_scores, axis = 1)

# Print the average scores
for i, train_size in enumerate(train_sizes): # Since it prints each training and testing score for each fold (each training size)
    print(f"Train Size: {train_size:.1f}, Average Training Score: {avg_train[i]:.2f}, Average Testing Score: {avg_test[i]:.2f}")
    
plt.figure(figsize = (10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis = 1), 'o-', label = 'Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis = 1), 'o-', label = 'Testing Score')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy Score')
plt.title('Learning Curves (Linear Regression)')
plt.legend(loc = 'best')
plt.grid(True)
plt.show()