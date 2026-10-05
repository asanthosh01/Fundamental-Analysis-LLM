"""Join audited sentiment, as-of fundamentals, and independent future-return labels."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from financial_analysis.features import structured_features
from financial_analysis.labels import forward_label
from financial_analysis.sentiment import chunks, validate_output

def prepare(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    manifest = pd.read_csv(manifest_path)
    required = {'ticker', 'filing_date', 'filing_text', 'companyfacts_json', 'sentiment_json',
        'prices_csv', 'price_source'}
    if not required.issubset(manifest.columns):
        raise ValueError(f'Missing manifest columns: {sorted(required-set(manifest.columns))}')
    records, audit = [], []
    for row in manifest.to_dict('records'):
        load = lambda col: (manifest_path.parent / row[col]).resolve()
        source_text = load('filing_text').read_text()
        sentiment = json.loads(load('sentiment_json').read_text())
        if sentiment.get('backend') != 'local_mistral_langchain':
            raise ValueError('Real training preparation requires local-Mistral output, not fixture scores')
        if sentiment['source_sha256'] != hashlib.sha256(source_text.encode()).hexdigest():
            raise ValueError('Sentiment does not correspond to filing text')
        passages = chunks(source_text)
        outputs = sentiment.get('chunks', [])
        if len(passages) != len(outputs):
            raise ValueError('Sentiment output does not cover the filing chunks')
        for passage, value in zip(passages, outputs):
            validate_output(value, passage)
        recomputed = sum(value['score']*len(passage) for passage,value in zip(passages,outputs))/sum(map(len,passages))
        if abs(recomputed-float(sentiment['sentiment_score'])) > 1e-9:
            raise ValueError('Aggregate sentiment score is inconsistent with chunk outputs')
        facts = json.loads(load('companyfacts_json').read_text())
        feature_record = structured_features(facts, row['filing_date'], sentiment['sentiment_score'])
        label = forward_label(pd.read_csv(load('prices_csv')), row['filing_date'])
        records.append({'ticker': row['ticker'], 'filing_date': row['filing_date'],
            **label,
            'price_source': row['price_source'], 'source_type': 'real_filings',
            **feature_record['features']})
        audit.append({'ticker': row['ticker'], 'source_sha256': sentiment['source_sha256'], **feature_record})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2))
    return records

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', default='data/training.csv')
    args = parser.parse_args()
    prepare(args.manifest, args.out)
