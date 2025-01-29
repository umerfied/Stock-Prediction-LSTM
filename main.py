from config import *
from data_loader import download_stock_data, str_to_datetime
from data_processing import df_to_windowed_df, windowed_as_np, prepare_data_splits
from model import create_lstm_model, make_recursive_predictions
from visualization import *
from tuning import *

def main():
    # Download and prepare data
    df = download_stock_data(TICKER, TIME_PERIOD, TIME_INTERVAL)
    plot_stock_data(df)
    
    # Create windowed dataset
    last_date = str(df.index[-1].date())
    fourth_date = str(df.index[3].date())
    windowed_df = df_to_windowed_df(df, fourth_date, last_date, n=WINDOW_SIZE)
    
    # Process data
    dates, X, final_matrix, scalar = windowed_as_np(windowed_df)
    
    # Scale final matrix
    num_samples = final_matrix.shape[0]
    final_matrix_2D = final_matrix.reshape(num_samples, 1)
    final_matrix_scaled_2D = scalar.fit_transform(final_matrix_2D)
    final_matrix_scaled = final_matrix_scaled_2D.reshape(num_samples)
    
    # Split data
    train_data, val_data, test_data = prepare_data_splits(
        dates, X, final_matrix_scaled, 
        train_split=TRAIN_SPLIT, 
        val_split=VAL_SPLIT
    )
    dates_train, X_train, y_train = train_data
    dates_val, X_val, y_val = val_data
    dates_test, X_test, y_test = test_data
    
    # Plot splits
    plot_train_val_test_split(dates_train, dates_val, dates_test, 
                             y_train, y_val, y_test)
    
    # Create and train model
    model = create_lstm_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)
    
    # Make predictions
    train_prediction = model.predict(X_train).flatten()
    val_prediction = model.predict(X_val).flatten()
    test_prediction = model.predict(X_test).flatten()
    
    # Unscale predictions
    unscaled_predictions = [
        scalar.inverse_transform(pred.reshape(-1, 1)).flatten()
        for pred in [train_prediction, val_prediction, test_prediction]
    ]
    
    unscaled_actuals = [
        scalar.inverse_transform(y.reshape(-1, 1)).flatten()
        for y in [y_train, y_val, y_test]
    ]
    
    # Plot predictions
    plot_predictions_vs_actual(
        [dates_train, dates_val, dates_test],
        unscaled_actuals,
        unscaled_predictions
    )
    
    # Make recursive predictions
    recursive_predictions, recursive_date = make_recursive_predictions(
        model, X_train, dates_val, dates_test
    )
    
    # Plot recursive predictions
    actual_recursive_values = np.concatenate([y_val, y_test])
    plot_recursive_predictions(recursive_date, actual_recursive_values, recursive_predictions, scaled=True)
    
    # Plot unscaled recursive predictions
    recursive_predictions_2D = np.array(recursive_predictions).reshape(-1, 1)
    unscaled_recursive_predictions = scalar.inverse_transform(recursive_predictions_2D).flatten()
    actual_recursive_values_2D = actual_recursive_values.reshape(-1, 1)
    unscaled_actual_recursive_values = scalar.inverse_transform(actual_recursive_values_2D).flatten()
    
    plot_recursive_predictions(
        recursive_date, 
        unscaled_actual_recursive_values, 
        unscaled_recursive_predictions, 
        scaled=False
    )

if __name__ == "__main__":
    main() 