import pandas as pd
import numpy as np
"""BATIN TAHA ÖNAL 220315086"""
"TEST FOR SIMPLE LINEAR REGRESSION"
def predict_simple_linear(X, slope, intercept):
    return slope * X + intercept
def simple_linear_regression_test(data, models, target_column):
    predictions = []
    for period in data['Month'].unique():
        period_data = data[data['Month'] == period]
        X = period_data.drop(columns=[target_column, 'Month']).values.flatten()
        if period in models:
            slope, intercept = models[period]
            period_predictions = predict_simple_linear(X, slope, intercept)
            predictions.extend(period_predictions)
        else:
            predictions.extend([np.nan] * len(X))
    return predictions
test_predictions_simple = simple_linear_regression_test(test_df, models, target_column='Target')
print("Simple Linear Regression Predictions on Test Data:", test_predictions_simple)
"************"
"************"
"************"
"TEST FOR POLYNOMIAL REGRESSION"
import numpy as np

def predict_polynomial(X, coeffs):
    degree = len(coeffs) - 1
    X_poly = np.array([X**i for i in range(degree + 1)]).T
    return X_poly @ coeffs

def polynomial_regression_test(data, models, target_column, period_length=30):
    predictions = []
    for (period_start, period_end), coeffs in models.items():
        period_data = data.iloc[period_start:period_end]
        X = period_data.drop(columns=[target_column]).values.flatten()
        period_predictions = predict_polynomial(X, coeffs)
        predictions.extend(period_predictions)
    return predictions
test_predictions_poly = polynomial_regression_test(test_df, models, target_column='Target', period_length=30)
print("Polynomial Regression Predictions on Test Data:", test_predictions_poly)
