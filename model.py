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
def Optuned_model():
    study = optuna.create_study(direction='minimized')
    study.optimize(objective, n_trials = 100)
    best_params = study.best_params
    best_model = Sequential([
        LSTM(best_params['n_units'], activation = 'Relu', input_shape(X_train.shape[1], X_train.shape[2])),
        Dropout(best_params['n_dropout']),
        Dense(1)
    ])
    best_model.compile(
        loss='mae',
        optimizers = keras.optimizers.Adam(learning_rate = best_params['lr']),
        metrics = ['mean_absolute_error']
    )
    best_model.fit(X_train,Y_train,epochs = 50, verbose = 0, batch_size =32)
    best_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print('Validation MAE:',mae)
    
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