from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

def create_lstm_model():
    """Create and return the LSTM model"""
    model = Sequential([
        Input((3, 1)),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.005),
        metrics=['mean_absolute_error']
    )
    
    return model

def make_recursive_predictions(model, X_train, dates_val, dates_test, noise_level=0.1):
    """Make recursive predictions"""
    from copy import deepcopy
    
    recursive_predictions = []
    recursive_date = np.concatenate([dates_val, dates_test])
    last_window = deepcopy(X_train[-1])

    for target_date in recursive_date:
        next_prediction = model.predict(np.array([last_window])).flatten()
        next_prediction += np.random.normal(scale=noise_level)
        recursive_predictions.append(next_prediction[0])
        last_window = np.roll(last_window, -1)
        last_window[-1] = (last_window[-1] + next_prediction[0])/2
    
    return recursive_predictions, recursive_date