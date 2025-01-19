# Business-Stock-Analysis
Business &amp; Stock Analysis

This is a Python-based Streamlit application that provides functionalities for analyzing stock market data. It supports comparing multiple stocks, viewing individual stock data, and visualizing market holidays. The app leverages various libraries such as `yfinance`, `plotly`, and `matplotlib` to offer a comprehensive stock analysis dashboard.

## Features

### 1. Stock Comparison
- Compare up to three stocks over a specified time period.
- Visualize and compare:
  - Closing prices
  - Trading volumes
  - Total traded values (Open Price x Volume)
- Calculate and display annual returns for the selected stocks.

### 2. Individual Stock Data
- View detailed stock information, including:
  - Price movements with annual return and risk metrics.
  - Fundamental data like balance sheets, income statements, and cash flow statements.
  - Top 10 news headlines with sentiment analysis using the `stocknews` library.
  - Moving averages (100-day and 200-day) visualizations.
- Predict future stock prices using a pre-trained Keras deep learning model.

### 3. Holidays
- List stock market holidays for a selected year (default is the current year).

## Prerequisites
Ensure the following Python libraries are installed:

```bash
pip install streamlit pandas yfinance plotly matplotlib holidays numpy keras scikit-learn stocknews
```

## Files
- `CodeList.xlsx`: An Excel file containing a list of stock ticker symbols with a column named `Yahoo Code`.
- `keras_model.h5`: A pre-trained Keras model for stock price prediction.

## How to Run the App
1. Clone the repository and navigate to the project directory.
2. Place the `CodeList.xlsx` and `keras_model.h5` files in the project folder.
3. Run the app using Streamlit:

   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser and explore the features.

## App Navigation
The app contains three main sections accessible via the sidebar:

### Stock Comparison
1. Select up to three stock tickers from the dropdown.
2. Specify the start and end dates for the analysis.
3. View visualizations for closing prices, trading volumes, and total traded values.
4. Annual returns for the selected stocks are displayed below the charts.

### Individual Stock Data
1. Enter a stock ticker and date range in the sidebar.
2. Explore tabs for various analyses:
   - **Pricing Data**: View price movements and annual metrics.
   - **Fundamental Data**: Check financial statements.
   - **Top 10 News**: See recent news with sentiment analysis.
   - **100MA & 200MA**: Visualize moving averages.
   - **Prediction Graph**: Predict future prices using the pre-trained model.

### Holidays
1. Enter a year in the sidebar to view a list of stock market holidays.

## Libraries Used
- `streamlit`: Web framework for building interactive apps.
- `pandas`: Data manipulation and analysis.
- `yfinance`: Downloading stock market data.
- `plotly`: Creating interactive visualizations.
- `matplotlib`: Plotting charts.
- `numpy`: Numerical computations.
- `holidays`: Fetching holiday information.
- `keras`: Deep learning model loading and predictions.
- `scikit-learn`: Data preprocessing for predictions.
- `stocknews`: Fetching news and sentiment analysis.

## Future Improvements
- Add error handling for invalid tickers or missing data.
- Enable custom configurations for moving averages.
- Improve the user interface for better navigation.
- Integrate real-time stock data updates.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as needed.
