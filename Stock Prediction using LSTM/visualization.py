import matplotlib.pyplot as plt

def plot_stock_data(df):
    """Plot initial stock data"""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Stock Price History')
    plt.show()

def plot_train_val_test_split(dates_train, dates_val, dates_test, y_train, y_val, y_test):
    """Plot data splits"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, y_train, label='train')
    plt.plot(dates_val, y_val, label='val', linestyle=':')
    plt.plot(dates_test, y_test, label='test', linestyle=":")
    plt.legend()
    plt.title('Data Splits')
    plt.show()

def plot_predictions_vs_actual(dates_train, dates_val, dates_test, 
                             actual_values, predicted_values, 
                             set_names=['Train', 'Validation', 'Test']):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for dates, actuals, name in zip([dates_train, dates_val, dates_test], 
                                  actual_values, set_names):
        plt.plot(dates, actuals, label=f'{name} Actual')
    plt.legend()
    plt.title('Actual Values')
    
    plt.subplot(1, 2, 2)
    for dates, preds, name in zip([dates_train, dates_val, dates_test], 
                                predicted_values, set_names):
        plt.plot(dates, preds, label=f'{name} Predicted')
    plt.legend()
    plt.title('Predicted Values')
    
    plt.tight_layout()
    plt.show()

def plot_recursive_predictions(recursive_date, actual_values, recursive_predictions, scaled=True):
    """Plot recursive predictions"""
    plt.figure(figsize=(12, 6))
    plt.plot(recursive_date, actual_values, label='Actual Prices')
    plt.plot(recursive_date, recursive_predictions, label='Recursive Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Recursive Predictions vs Actual Prices {"(Scaled)" if scaled else "(Unscaled)"}')
    plt.legend()
    plt.show()