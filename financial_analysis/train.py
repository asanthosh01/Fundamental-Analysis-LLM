"""Train a return-direction XGBoost classifier with purged chronological splits."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
from financial_analysis.features import FEATURES

def validate(data):
    required = set(FEATURES) | {'ticker', 'filing_date', 'label_end_date', 'forward_return', 'source_type', 'price_source'}
    if not required.issubset(data.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(data.columns))}')
    data = data.copy()
    if data[list(required)].isna().any().any():
        raise ValueError('Missing feature or provenance values')
    for col in ['filing_date', 'label_end_date']:
        data[col] = pd.to_datetime(data[col], errors='raise').dt.normalize()
    if (data.label_end_date <= data.filing_date).any():
        raise ValueError('Return labels must end after the filing date')
    if data.duplicated(['ticker', 'filing_date']).any():
        raise ValueError('Duplicate company/filing dates')
    for col in FEATURES + ['forward_return']:
        data[col] = pd.to_numeric(data[col], errors='raise')
    if not np.isfinite(data[FEATURES + ['forward_return']].values).all():
        raise ValueError('Features and labels must be finite')
    if not data.sentiment_score.between(-1, 1).all():
        raise ValueError('Sentiment scores must be between -1 and 1')
    if not set(data.source_type).issubset({'synthetic_demo', 'real_filings'}) or data.source_type.nunique() != 1:
        raise ValueError('Use only real_filings or only synthetic_demo, never mixed data')
    data['positive_outlook'] = (data.forward_return > 0).astype(int)
    return data.sort_values(['filing_date', 'ticker']).reset_index(drop=True)

def split(data):
    days = sorted(data.filing_date.unique())
    if len(data) < 50 or len(days) < 10:
        raise ValueError('Need at least 50 rows across 10 dates')
    val_start = pd.Timestamp(days[int(len(days)*.6)])
    test_start = pd.Timestamp(days[int(len(days)*.8)])
    # A training label cannot use a price from inside a later split.
    train = data[(data.filing_date < val_start) & (data.label_end_date < val_start)]
    val = data[(data.filing_date >= val_start) & (data.filing_date < test_start) & (data.label_end_date < test_start)]
    test = data[data.filing_date >= test_start]
    if any(len(x) < 5 for x in [train, val, test]):
        raise ValueError('Insufficient rows after purging overlapping label horizons')
    return train, val, test

def metrics(y, probability):
    return {'n': len(y), 'accuracy': float(accuracy_score(y, probability >= .5)),
        'balanced_accuracy': float(balanced_accuracy_score(y, probability >= .5)),
        'brier_score': float(brier_score_loss(y, probability)),
        'roc_auc': float(roc_auc_score(y, probability)) if y.nunique() == 2 else None}

def train(data_path, out):
    data_path, out = Path(data_path), Path(out)
    data = validate(pd.read_csv(data_path))
    parts = split(data)
    if parts[0].positive_outlook.nunique() != 2:
        raise ValueError('Training split must contain both target classes')
    model = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=.05,
        subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=2, eval_metric='logloss')
    model.fit(parts[0][FEATURES], parts[0].positive_outlook)
    prior = float(parts[0].positive_outlook.mean())
    report = {'schema_version': 1, 'source_type': data.source_type.iloc[0],
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(), 'features': FEATURES,
        'target': 'positive when supplied post-filing forward_return > 0, negative otherwise',
        'method': '60/20/20 chronological dates; purge training/validation labels crossing the next split; fixed parameters',
        'rows': len(data), 'purged_rows': len(data)-sum(map(len, parts)),
        'probabilities': 'Raw model estimates, not calibrated', 'splits': {},
        'feature_importance': dict(zip(FEATURES, map(float, model.feature_importances_)))}
    for name, part in zip(['train', 'validation', 'test'], parts):
        report['splits'][name] = {'start': str(part.filing_date.min().date()),
            'end': str(part.filing_date.max().date()),
            'model': metrics(part.positive_outlook, model.predict_proba(part[FEATURES])[:, 1]),
            'majority_baseline': metrics(part.positive_outlook, np.full(len(part), prior))}
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out/'model.json')
    (out/'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'source_type': report['source_type'], 'test': report['splits']['test']}, indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/training.csv')
    parser.add_argument('--out', default='artifacts')
    args = parser.parse_args()
    train(args.data, args.out)
