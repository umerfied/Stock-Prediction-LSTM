# Stock Price Prediction using LSTM
A comprehensive guide to implementing and using the LSTM-based stock price prediction model.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Module Details](#module-details)
6. [Usage](#usage)
7. [Model Architecture](#model-architecture)
8. [Output](#output)
9. [Troubleshooting](#troubleshooting)

## Overview
This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical data. The system is capable of:
- Downloading historical stock data
- Processing time-series data for LSTM input
- Training an LSTM model
- Making both direct and recursive predictions
- Visualizing results

## Project Structure Stock Prediction using LSTM/
├── main.py # Main execution script
├── config.py # Configuration parameters
├── data_loader.py # Stock data downloading functions
├── data_processing.py # Data preprocessing utilities
├── model.py # LSTM model definition
├── visualization.py # Plotting functions
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Stock-Prediction-LSTM
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow >= 2.0.0
- pandas >= 1.0.0
- numpy >= 1.18.0
- yfinance >= 0.1.63
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- datetime

## Configuration
The `config.py` file contains all configurable parameters:

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| TICKER | Stock symbol | 'AAPL' |
| TIME_PERIOD | Historical data period | '2y' |
| TIME_INTERVAL | Data frequency | '1d' |
| WINDOW_SIZE | Time steps for LSTM | 60 |
| TRAIN_SPLIT | Training data proportion | 0.7 |
| VAL_SPLIT | Validation data proportion | 0.15 |

## Module Details

### main.py
The main script that orchestrates the entire prediction process:
- Data downloading and preparation
- Model creation and training
- Making predictions
- Visualization of results

### data_loader.py
Handles data acquisition:
- Downloads stock data using yfinance
- Converts date strings to datetime objects
- Manages data retrieval errors

### data_processing.py
Processes raw data for model input:
- Creates windowed datasets
- Converts data to numpy arrays
- Handles data scaling
- Splits data into training/validation/test sets

### model.py
Defines the LSTM model architecture:
- Creates LSTM layers
- Configures model parameters
- Implements prediction functions
- Handles recursive predictions

### visualization.py
Provides visualization functions:
- Stock price history plots
- Training/validation/test split visualization
- Prediction vs actual comparisons
- Recursive prediction plots

## Usage

### Basic Usage
Run the model with default configuration:
```bash
python main.py
```

### Custom Configuration
1. Modify `config.py` with desired parameters
2. Run the main script
3. Check generated visualizations in the output

## Model Architecture

### LSTM Structure
- Input Layer: Shape based on window size
- LSTM Layers: Multiple stacked layers
- Dense Layers: For final prediction
- Output: Single value prediction

### Training Parameters
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 200 (default)
- Validation: Using validation split

## Output

### Generated Visualizations
1. Historical Stock Data
   - Raw price data
   - Trading volume

2. Data Split Visualization
   - Training set
   - Validation set
   - Test set

3. Predictions
   - Direct predictions vs actual values
   - Recursive predictions
   - Scaled and unscaled versions

### Prediction Types
1. Direct Predictions
   - One-step-ahead forecasting
   - Uses actual values as input

2. Recursive Predictions
   - Multi-step forecasting
   - Uses previous predictions as input

## Troubleshooting

### Common Issues

1. Data Download Errors
   - Check internet connection
   - Verify stock ticker symbol
   - Ensure yfinance API is working

2. Memory Issues
   - Reduce window size
   - Decrease data period
   - Use smaller batch size

3. Training Problems
   - Adjust learning rate
   - Modify model architecture
   - Check for data normalization issues

### Best Practices
1. Start with small data samples
2. Validate data preprocessing
3. Monitor training metrics
4. Use appropriate scaling

## Disclaimer
This project is for educational purposes only. The predictions should not be used as financial advice.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please feel free to submit pull requests.