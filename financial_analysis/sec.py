"""Download SEC JSON with explicit contact identity and bounded requests."""
import argparse
import json
import os
import time
from pathlib import Path
import requests

def fetch(cik, kind, destination, user_agent):
    if not user_agent or '@' not in user_agent:
        raise ValueError('Set SEC_USER_AGENT to your name and real contact email')
    if not str(cik).isdigit() or len(str(cik)) > 10:
        raise ValueError('CIK must be a number of at most 10 digits')
    if kind not in {'companyfacts', 'submissions'}:
        raise ValueError('kind must be companyfacts or submissions')
    padded = str(cik).zfill(10)
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{padded}.json' if kind == 'companyfacts' else f'https://data.sec.gov/submissions/CIK{padded}.json'
    response = requests.get(url, headers={'User-Agent': user_agent, 'Accept': 'application/json'}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    time.sleep(.25)
    return payload

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cik', required=True)
    parser.add_argument('--kind', choices=['companyfacts', 'submissions'], default='companyfacts')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    fetch(args.cik, args.kind, args.out, os.environ.get('SEC_USER_AGENT'))
