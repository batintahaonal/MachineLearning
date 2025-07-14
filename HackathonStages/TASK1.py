import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""BATIN TAHA ÖNAL 220315086"""
def simple_linear_regression(X, y):
    n = len(X)
    mean_x, mean_y = np.mean(X), np.mean(y)
    numerator = np.sum((X - mean_x) * (y - mean_y))
    denominator = np.sum((X - mean_x) ** 2)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept
def predict(X, slope, intercept):
    return slope * X + intercept
def monthly_regression_model(data, target_column, periods):
    models = {}
    predictions = pd.DataFrame(index=data.index, columns=periods)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(periods)))
    for idx, period in enumerate(periods):
        period_data = data[data['Month'] == period]
        X = period_data.drop(columns=[target_column, 'Month']).values.flatten()
        y = period_data[target_column].values

        # Train regression model for each period
        slope, intercept = simple_linear_regression(X, y)
        models[period] = (slope, intercept)

        predictions.loc[period_data.index, period] = predict(X, slope, intercept)
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = predict(x_line, slope, intercept)
        plt.plot(x_line, y_line, color=colors[idx], label=f'Regression line for {period}')
    combined_predictions = predictions.mean(axis=1)
    mse = np.mean((data[target_column] - combined_predictions) ** 2)
    print(f"Mean Squared Error of Combined Model: {mse}")
    plt.scatter(data.drop(columns=[target_column, 'Month']).values.flatten(),
                data[target_column].values, color='black', label='Data Points', alpha=0.5)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Regression Lines for Each Period")
    plt.legend()
    plt.show()
    return combined_predictions, models
