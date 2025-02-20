#pip install pandas numpy scikit-learn hyperopt
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import pickle
import warnings
warnings.filterwarnings("ignore")

# Step 2: Load the dataset
data = pd.read_csv(r"C:\Users\megha\Downloads\ENB2012_data.csv")

# Basic structure #Understand Dataset Structure
print(data.shape)  # (rows, columns)
print(data.dtypes)  # Data types of each column
print(data.head())  # First few rows

# Check for missing values
print("Missing Values Before Handling:")
print(data.isnull().sum())

# Fill missing numerical values with the mean
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mean(), inplace=True)

# Fill missing categorical values with the mode
for col in data.select_dtypes(include=['object']).columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Verify that missing values are handled
print("Missing Values After Handling:")
print(data.isnull().sum())

 # Summary statistics
print(data.describe())

# Identify Missing Values
print(data.isnull().sum())  # Count of missing values per column

#Check for Duplicates
print(data.duplicated().sum())  # Number of duplicate rows
data = data.drop_duplicates()  # Remove duplicates

#Data Distribution
import matplotlib.pyplot as plt
data.hist(figsize=(12, 10))  # Histogram for all numerical features
plt.show()

# Standardization
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_standard = StandardScaler()
standardized_data = scaler_standard.fit_transform(data)

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
normalized_data = scaler_minmax.fit_transform(data)

# Convert back to DataFrame for interpretability
standardized_df = pd.DataFrame(standardized_data, columns=data.columns)
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

print("Standardized Data:\n", standardized_df)
print("Normalized Data:\n", normalized_df)

#Correlation Analysis
import seaborn as sns
# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#Categorical vs Numerical Features
categorical_cols = data.select_dtypes(include=['int64', 'object']).columns
numerical_cols = data.select_dtypes(include=['float64']).columns
print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# Outlier Detection
# Boxplot for each feature
for col in data.columns:
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Visualize target variable distributions
sns.histplot(data['Y1'], kde=True)
plt.title("Distribution of Y1")
plt.show()

sns.histplot(data['Y2'], kde=True)
plt.title("Distribution of Y2")
plt.show()

df = pd.DataFrame(data)

# Step 3: Define the features (X) and target (Y1)
X = df.drop(columns=["Y1", "Y2"])
y = df[["Y1","Y2"]]

# Step 4: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the search space for models and hyperparameters
space = hp.choice('regressor_type', [
    {
        'model': 'RandomForest',
        'n_estimators': scope.int(hp.quniform('rf_n_estimators', 50, 500, 10)),
        'max_depth': scope.int(hp.quniform('rf_max_depth', 3, 20, 1))
    },
    {
        'model': 'GradientBoosting',
        'n_estimators': scope.int(hp.quniform('gb_n_estimators', 50, 500, 10)),
        'learning_rate': hp.uniform('gb_learning_rate', 0.01, 0.3),
        'max_depth': scope.int(hp.quniform('gb_max_depth', 3, 20, 1))
    }
])

# Step 5: Define the objective function for Bayesian Optimization
def objective(params):
    if params['model'] == 'RandomForest':
        model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=42
        ))
    elif params['model'] == 'GradientBoosting':
        model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=int(params['max_depth']),
            random_state=42
        ))

    # Perform cross-validation and return RMSE
    predictions = cross_val_predict(model, X_train, y_train, cv=5)
    rmse = np.sqrt(mean_squared_error(y_train, predictions, multioutput='uniform_average'))
    return {'loss': rmse, 'status': STATUS_OK}

# Step 6: Run Bayesian Optimization
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# Step 7: Train and evaluate the best model
if best_params['regressor_type'] == 0:
    final_model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=int(best_params['rf_n_estimators']),
        max_depth=int(best_params['rf_max_depth']),
        random_state=42
    ))
elif best_params['regressor_type'] == 1:
    final_model = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=int(best_params['gb_n_estimators']),
        learning_rate=best_params['gb_learning_rate'],
        max_depth=int(best_params['gb_max_depth']),
        random_state=42
    ))

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Step 8: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Best Parameters Found: ", best_params)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2 Score: {r2:.4f}")

# Make pickle of our model
pickle.dump(final_model, open("model.pkl", "wb"))

