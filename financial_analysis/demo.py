"""Synthetic smoke-test data; never report these scores as market performance."""
from pathlib import Path
import numpy as np
import pandas as pd
from financial_analysis.train import train

def generate(path, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for period, day in enumerate(pd.date_range('2016-01-01', periods=48, freq='MS')):
        for company in range(5):
            sentiment = rng.uniform(-1, 1)
            growth = rng.normal(.04, .12)
            margin = rng.normal(.1, .08)
            cash = rng.normal(.12, .06)
            liabilities = rng.uniform(.2, .9)
            # Deliberately simulated relationship; does not establish a real edge.
            outcome = .035*sentiment + .2*growth + .1*margin -.02*liabilities + rng.normal(0,.06)
            rows.append({'ticker': f'DEMO{company}', 'filing_date': day.date().isoformat(),
                'label_end_date': (day+pd.Timedelta(days=28)).date().isoformat(),
                'forward_return': outcome, 'source_type': 'synthetic_demo',
                'price_source': 'seeded synthetic generator; not market prices',
                'sentiment_score': sentiment, 'revenue_growth': growth, 'net_margin': margin,
                'cashflow_margin': cash, 'liabilities_to_assets': liabilities})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

if __name__ == '__main__':
    generate('data/demo_training.csv')
    train('data/demo_training.csv', 'artifacts/demo')
