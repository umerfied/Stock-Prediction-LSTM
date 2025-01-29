import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    
	first_date = str_to_datetime(first_date_str)
	last_date = str_to_datetime(last_date_str)

	target_date = first_date
	last_time = False
	dates = []
	X, Y = [], []

	while True:
     
		df_subset = dataframe.loc[:target_date].tail(n+1)

		if len(df_subset) != n+1:
			print(f'Error: Window of size {n} is too large for date {target_date}')
			break

		values = df_subset['Close'].to_numpy()
		x, y = values[:-1], values[-1]

		dates.append(target_date)
		X.append(x)
		Y.append(y)

		# Use timedelta like this:
		next_week = dataframe.loc[target_date:target_date+timedelta(days=7)]
		next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
		next_date_str = next_datetime_str.split('T')[0]
		year_month_day = next_date_str.split('-')
		year, month, day = year_month_day
		next_date = datetime(day=int(day), month=int(month), year=int(year))
		if last_time:
			break	
		target_date = next_date

		if target_date == last_date:
			last_time = True
		ret_df = pd.DataFrame({})
		ret_df['Target Date'] = dates
					
		X = np.array(X)
		for i in range(0, n):
			X[:, i]
			ret_df[f'Target-{n-i}'] = X[:, i]
			ret_df['Target'] = Y
	return ret_df

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