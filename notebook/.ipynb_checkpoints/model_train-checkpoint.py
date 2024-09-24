import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate features and target for training data
X_train = train_data.drop('Sales', axis=1)
y_train = train_data['Sales']

# Define the pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('rf_regressor', RandomForestRegressor(random_state=42))
])

# Define the parameter grid
param_grid = {
    'imputer__strategy': ['mean', 'median'],
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
feature_importances = best_model.named_steps['rf_regressor'].feature_importances_
feature_names = X_train.columns
for name, importance in zip(feature_names, feature_importances):
    print(f"Feature '{name}': {importance}")