import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""BATIN TAHA ÖNAL 220315086"""
def polynomial_regression(X, y, degree):
    X_poly = np.array([X ** i for i in range(degree + 1)]).T
    coeffs = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    return coeffs
def predict(X, coeffs):
    degree = len(coeffs) - 1
    X_poly = np.array([X ** i for i in range(degree + 1)]).T
    return X_poly @ coeffs
def monthly_polynomial_regression_model(data, target_column, degree, period_length=30):
    models = {}
    predictions = []
    plt.figure(figsize=(12, 6))
    for period_start in range(0, len(data), period_length):
        period_end = min(period_start + period_length, len(data))
        period_data = data.iloc[period_start:period_end]
        X = period_data.drop(columns=[target_column]).values.flatten()
        y = period_data[target_column].values

        coeffs = polynomial_regression(X, y, degree)
        models[(period_start, period_end)] = coeffs
        period_predictions = predict(X, coeffs)
        predictions.extend(period_predictions)
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = predict(x_line, coeffs)
        plt.plot(x_line, y_line, label=f'Polynomial fit for days {period_start}-{period_end}')
    mse = np.mean((data[target_column].values - predictions) ** 2)
    print(f"Mean Squared Error of Combined Model: {mse}")
    plt.scatter(data.drop(columns=[target_column]).values.flatten(), data[target_column].values, color='black',
                alpha=0.5, label="Data Points")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Polynomial Regression Lines for Each 30-Day Period")
    plt.legend()
    plt.show()
    return predictions, models
predictions, models = monthly_polynomial_regression_model(df, target_column='Target', degree=degree, period_length=30)
