import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#Encode categorical variables
from sklearn.preprocessing import LabelEncoder
#Split the Dataset into Training and Testing Sets
from sklearn.model_selection import train_test_split
#Train a Linear Regression Model
from sklearn.linear_model import LinearRegression
#make predictions
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#Random Forest Regressor:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib




# Load the dataset
file_path = "car data.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Preview the first 5 rows of the dataset
#print(df.head())

# Get a summary of the dataset
#print(df.info())

# Get summary statistics for numerical columns
#print(df.describe())

# Check for missing values
#print(df.isnull().sum())

# Get column types
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

#print("Categorical Columns:", categorical_columns)
#print("Numerical Columns:", numerical_columns)

# Select only numerical columns for correlation
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Compute correlation matrix
correlation_matrix = numerical_df.corr()

# Visualize the correlation matrix as a heatmap
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
#plt.title("Correlation Matrix")
#plt.show()

# Distribution of car prices
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.histplot(df['Selling_Price'], kde=True, bins=30)
#plt.xlabel("Selling Price")
#plt.ylabel("Frequency")
#plt.title("Selling Price Distribution")
#plt.show()

# Car Age vs Selling Price
# Calculate car age from the 'Year' column
#df['Car_Age'] = 2024 - df['Year']  # Assuming the current year is 2024
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.scatterplot(x='Car_Age', y='Selling_Price', data=df)
#plt.title('Car Age vs Selling Price')
#plt.xlabel('Car Age (years)')
#plt.ylabel('Selling Price')
#plt.show()

#Present Price vs Selling Price
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.scatterplot(x='Present_Price', y='Selling_Price', data=df)
#plt.title('Present Price vs Selling Price')
#plt.xlabel('Present Price')
#plt.ylabel('Selling Price')
#plt.show()

# Scatter plot: Price vs. Mileage (replace 'Price' and 'Mileage' with actual column names)
#sns.set(style="whitegrid")
#sns.scatterplot(data=df, x='Driven_kms', y='Selling_Price')
#plt.title("Price vs. Mileage")
#plt.show()

#Data Preprocessing
# Check for missing values
#print("Missing Values in Each Column:")
#print(df.isnull().sum())


label_encoder = LabelEncoder()

# Encode 'Fuel_Type'
df['Fuel_Type'] = label_encoder.fit_transform(df['Fuel_Type'])

# Encode 'Seller_Type'
df['Selling_type'] = label_encoder.fit_transform(df['Selling_type'])

# Encode 'Transmission'
df['Transmission'] = label_encoder.fit_transform(df['Transmission'])

# 'Car_Name' may not be relevant for prediction, but let's keep it encoded
df['Car_Name'] = label_encoder.fit_transform(df['Car_Name'])

# Define features and target variable
X = df.drop(['Selling_Price'], axis=1)  # Features
y = df['Selling_Price']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("Training Set Shape:", X_train.shape)
#print("Testing Set Shape:", X_test.shape)


# Initialize the model
#linear_regressor = LinearRegression()

# Train the model
#linear_regressor.fit(X_train, y_train)

#print("Linear Regression model training completed.")


# Initialize the scaler
#scaler = StandardScaler()

# Fit and transform the training data
#X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data (we do not fit the scaler on the test data, only transform)
#X_test_scaled = scaler.transform(X_test)

# Re-train the model with scaled data
#linear_regressor.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
#y_pred_scaled = linear_regressor.predict(X_test_scaled)





#Random Forest Regressor:
# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on scaled training data
rf_model.fit(X_train, y_train)
#print("Random Forest Regressor model training completed.")

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Compare actual vs. predicted values
comparison_scaled = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
#print(comparison_scaled.head())

# Setting a style for all plots
#sns.set(style="whitegrid")
# Scatter plot of actual vs. predicted prices
#plt.figure(figsize=(8, 6))
#plt.scatter(y_test, y_pred_rf, color='blue')
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
#plt.xlabel('Actual Prices')
#plt.ylabel('Predicted Prices')
#plt.title('Actual vs Predicted Prices')
#plt.show()

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Display the results
#print(f"MAE (Random Forest): {mae_rf}")
#print(f"MSE (Random Forest): {mse_rf}")
#print(f"RMSE (Random Forest): {rmse_rf}")
#print(f"R² Score (Random Forest): {r2_rf}")

joblib.dump(rf_model, "car_price_model.pkl")
# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a bar chart
#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#plt.barh(feature_names, importances, color="skyblue")
#plt.xlabel("Feature Importance")
#plt.ylabel("Features")
#plt.title("Feature Importance in Predicting Car Prices")
#plt.show()