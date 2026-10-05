"""Independent return labels from adjusted daily closing prices."""
import numpy as np
import pandas as pd

def forward_label(prices, filing_date, horizon=20):
    if not {'date', 'adjusted_close'}.issubset(prices.columns):
        raise ValueError('Price CSV requires date and adjusted_close columns')
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Horizon must be a positive number of trading sessions')
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date'], errors='raise').dt.normalize()
    prices['adjusted_close'] = pd.to_numeric(prices['adjusted_close'], errors='raise')
    if prices.date.isna().any() or prices.date.duplicated().any() or not np.isfinite(prices.adjusted_close).all() or (prices.adjusted_close <= 0).any():
        raise ValueError('Prices must have unique dates and finite positive values')
    future = prices[prices.date > pd.Timestamp(filing_date)].sort_values('date').reset_index(drop=True)
    if len(future) <= horizon:
        raise ValueError('Not enough post-filing prices for the requested horizon')
    entry, end = future.iloc[0], future.iloc[horizon]
    return {'label_start_date': str(entry.date.date()), 'label_end_date': str(end.date.date()),
        'forward_return': float(end.adjusted_close/entry.adjusted_close-1)}
