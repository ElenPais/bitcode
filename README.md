# BitCode: Machine Learning for Stock Predictions
BitCode applies Machine Learning to predict and enhance the performance of Bitcoin, Apple, and Amazon stock transactions, enabling investors to make smarter, data-driven decisions with historical and real-time insights.

## Project Overview
- **Goal**: Predict stock price movements for Bitcoin, Apple, and Amazon using Machine Learning models.
- **Features**:
  - Predict stock price trends based on historical data.
  - Real-time stock prediction using live data.
  - User-friendly dashboard for visualizing stock data and predictions using Plotly and Dash.

## Libraries and Dependencies
This project uses the following libraries:

- **Python**: 3.10.10
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For implementing machine learning models like RandomForestRegressor and data preprocessing.
- **Dash**: For building interactive web applications and dashboards.
- **Matplotlib**: For data visualization.
- **Plotly**: For interactive plots and charts.
- **Joblib**: For saving and loading the trained machine learning models.
- **Datetime**: For handling date-related operations.

## How it Works
### 1. Data Loading:
- The stock price data for Bitcoin, Apple, and Amazon is loaded from CSV files.
- The data is preprocessed, with a new column indicating the source (Bitcoin, Apple, or Amazon).

### 2. Machine Learning Models:
- Machine learning models are trained for each stock using historical stock data.
- The models are saved using `joblib` and used to make real-time stock predictions.

### 3. Visualization:
Data visualizations are created using Plotly, including:
- Growth rate of stocks over the years.
- Monthly volume of trades.
- Predicted vs. actual stock values.

### 4. Dashboard:
The interactive Dash dashboard allows users to visualize the data and select different stocks (Bitcoin, Apple, Amazon) to see various charts and predictions.

## Requirements
To run the project, you will need to have the following libraries installed:

- **Python 3.10.10**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Dash**
- **Plotly**
- **Matplotlib**
- **Joblib**
- **Datetime**
