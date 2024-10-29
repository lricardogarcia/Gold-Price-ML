# Gold Price Analysis and Prediction

This project leverages machine learning, specifically a Random Forest Regressor model, to predict the price of gold (GLD) based on a dataset containing various financial indicators. The dataset includes information on the gold price alongside other significant financial indices, such as SPX, USO, SLV, and EUR/USD, over time.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Data Exploration and Visualization](#data-exploration-and-visualization)
  - [Correlation Analysis](#correlation-analysis)
  - [GLD Price Distribution](#gld-price-distribution)
- [Model Development and Training](#model-development-and-training)
  - [Data Splitting](#data-splitting)
  - [Random Forest Model Training](#random-forest-model-training)
- [Evaluation and Results](#evaluation-and-results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Project Overview

This project aims to predict gold prices based on historical data. By using a Random Forest Regressor, the model is trained to understand the relationship between the gold price (GLD) and other financial variables, helping to provide accurate price forecasts.

## Technologies Used

- **Python** for data processing and model training
- **Pandas** and **NumPy** for data handling and manipulation
- **Matplotlib** and **Seaborn** for data visualization
- **Scikit-Learn** for machine learning model implementation

## Dataset

The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data). It contains 2,290 rows and 7 columns, with the following key features:
- **Date**: The date for each data point
- **SPX, GLD, USO, SLV, EUR/USD**: Key financial indicators with GLD representing gold price
- **Rows and Columns**: 2,290 rows and 7 columns

## Data Exploration and Visualization

The data is analyzed to gain insights before training the model.

### Correlation Analysis
A heatmap is generated to visualize the correlation between different financial variables and the gold price. This allows us to understand how each variable relates to GLD. 

```python
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
```

### GLD Price Distribution
The distribution of the GLD price is visualized to understand its statistical properties and how it varies across the dataset.

```python
sns.displot(gold_data['GLD'], color='green')
```

## Model Development and Training

### Data Splitting
The dataset is split into training and testing sets with an 80-20 split. The target variable is GLD, while the features include SPX, USO, SLV, and EUR/USD.

### Random Forest Model Training
The Random Forest Regressor model is trained to predict the gold price. Here’s the training process:

```python
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)
```

## Evaluation and Results

The model’s performance is evaluated using the R-squared metric, which measures the accuracy of the predictions on the test set.

```python
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)
```

A comparison plot is also created to visualize actual vs. predicted values of the gold price, as shown below:

```python
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gold-price-prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the script to load and analyze the data, train the model, and visualize results.
   ```bash
   python gold_price_prediction.py
   ```

## References

- Kaggle Dataset: [Gold Price Data](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data)
