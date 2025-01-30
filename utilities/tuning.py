import optuna
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_processing import *
from core_modules.model import create_lstm_model
from core_modules.config import *

def objective(trials):
    n_dropout = trials.suggest_float('n_dropout', 0.1, 0.5, log = True)
    lr = trials.suggest_float('lr', 1e-5,1e-1)
    n_units = trials.suggest_int('n_units', 32, 128)
    
    model.Sequential([
        LSTM(n_units, activation = 'Relu',input_shape(X_train.shape[1],X_train.shape[2])),
        Dropout(n_dropout),
        Dense(1)
    ])
    model.compile(
        loss='mae',
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        metrics = ['mean_absolute_error']
    )
    model.fit(X_train, y_train,epochs = 50, verbose = 0, batch_size = 32)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(X_val, y_val)
    return mae
    
    