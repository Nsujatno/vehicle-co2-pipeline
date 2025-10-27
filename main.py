# installations
# pip install pandas numpy scikit-learn

# imports
import pandas as pd
import numpy as np

import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# 
# loading data
data = pd.read_csv("vehicle_emissions.csv") 
print(data.info())

# create features and target variables
X = data.drop(["CO2_Emissions"], axis=1)
y = data["CO2_Emissions"]

# split categorical and numerical features
numerical_cols = ["Model_Year", "Engine_Size", "Cylinders", "Fuel_Consumption_in_City(L/100 km)", "Fuel_Consumption_in_City_Hwy(L/100 km)", "Fuel_Consumption_comb(L/100km)", "Smog_Level"]
categorical_cols = ["Make", "Model", "Vehicle_Class", "Transmission"]

# start the pipeline with encoding
numerical_pipline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# join the pipelines together
preprocessor = ColumnTransformer([
    ('num', numerical_pipline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# split into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train and predict model
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)

# view the encoding that was done
# encoded_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
# print(encoded_cols)
# eval model accuracy

# the lower the better
mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)

# the higher the better
r2 = r2_score(y_test, prediction)

# lower score the better
mae = mean_absolute_error(y_test, prediction)

print(f'Model performance: ')
print(f'R2 Score: {r2}')
print(f'Root mean square error: {rmse}')
print(f'Mean absolute error: {mae}')

joblib.dump(pipeline, "vehicle_emission_pipeline.joblib")