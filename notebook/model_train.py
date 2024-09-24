import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

# Separate features and target for training data
X_train = train_data.drop('Sales', axis=1)
y_train = train_data['Sales']

# Identify numeric and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object', 'datetime64']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Define the full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf_regressor', RandomForestRegressor(random_state=42))
])

# Define the parameter grid
param_grid = {
    'rf_regressor__n_estimators': [100, 200, 300],
    'rf_regressor__max_depth': [None, 10, 20, 30],
    'rf_regressor__min_samples_split': [2, 5, 10]
}

# Set up the grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", -grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
test_predictions = best_model.predict(test_data)

# If you need to save the predictions
test_data['Predicted_Sales'] = test_predictions
test_data.to_csv('test_predictions.csv', index=False)

# Print feature importances
# Note: Feature names will be different due to one-hot encoding
feature_importances = best_model.named_steps['rf_regressor'].feature_importances_
feature_names = (numeric_features.tolist() + 
                 best_model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names(categorical_features).tolist())
for name, importance in zip(feature_names, feature_importances):
    print(f"Feature '{name}': {importance}")

from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error

huber = HuberRegressor()
huber.fit(X_train, y_train)

y_pred = huber.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
