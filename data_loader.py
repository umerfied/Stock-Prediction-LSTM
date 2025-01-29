import yfinance as yf
import pandas as pd
from datetime import datetime

def download_stock_data(ticker, period, interval):
    """Download stock data using yfinance"""
    data = yf.download(tickers=ticker, period=period, interval=interval)
    df = pd.DataFrame(data)
    
    # Clean up column names
    df.columns = df.columns.get_level_values(0).unique().tolist()
    
    # Process the dataframe
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df = df.set_index('Date')
    
    return df

def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime(year=year, month=month, day=day) 