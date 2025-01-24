import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    """Convert dataframe to windowed format"""
    # ... existing windowed df function ...
    # (Keep the entire function as is, just move it here)

def windowed_as_np(temp_windowed):
    """Convert windowed dataframe to numpy arrays"""
    scalar = MinMaxScaler()
    df_as_np = temp_windowed.to_numpy()
    dates = df_as_np[:,0]

    middle_matrix = df_as_np[:, 1:-1]
    scalar.fit(middle_matrix)
    middle_matrix = scalar.transform(middle_matrix)
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    final_matrix = df_as_np[:, -1]
    
    return dates, X.astype(np.float32), final_matrix.astype(np.float32), scalar

def prepare_data_splits(dates, X, final_matrix, train_split=0.8, val_split=0.9):
    """Split data into train, validation and test sets"""
    q_80 = int(len(dates) * train_split)
    q_90 = int(len(dates) * val_split)
    
    dates_train, X_train, y_train = dates[:q_80], X[:q_80], final_matrix[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], final_matrix[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], final_matrix[q_90:]
    
    return (dates_train, X_train, y_train), (dates_val, X_val, y_val), (dates_test, X_test, y_test) 