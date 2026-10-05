"""Financial ratios from SEC Company Facts, selected as-of the filing date."""
from datetime import date
import math

CONCEPTS = {
    'revenue': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues', 'SalesRevenueNet'],
    'net_income': ['NetIncomeLoss'], 'assets': ['Assets'],
    'liabilities': ['Liabilities'], 'operating_cashflow': ['NetCashProvidedByUsedInOperatingActivities']}
FEATURES = ['sentiment_score', 'revenue_growth', 'net_margin', 'cashflow_margin', 'liabilities_to_assets']

def select_fact(companyfacts, names, as_of, duration=False, before_end=None):
    as_of = date.fromisoformat(as_of)
    candidates = []
    for name in names:
        records = companyfacts.get('facts', {}).get('us-gaap', {}).get(name, {}).get('units', {}).get('USD', [])
        for fact in records:
            try:
                filed = date.fromisoformat(fact['filed'])
                end = date.fromisoformat(fact['end'])
                value = float(fact['val'])
                if filed > as_of or end > as_of or fact.get('form') not in {'10-K', '10-K/A'} or not math.isfinite(value):
                    continue
                if before_end and end >= date.fromisoformat(before_end):
                    continue
                if duration:
                    length = (end-date.fromisoformat(fact['start'])).days
                    if not 330 <= length <= 400:
                        continue
                candidates.append((end, filed, fact, name))
            except (KeyError, ValueError, TypeError):
                continue
        # Prefer a consistent concept over mixing different revenue definitions.
        if candidates:
            break
    if not candidates:
        raise ValueError(f'No eligible annual USD fact for {names} as of {as_of}')
    _, _, fact, concept = max(candidates, key=lambda x: (x[0], x[1]))
    return {**fact, 'concept': concept}

def structured_features(companyfacts, filing_date, sentiment_score):
    current = {}
    for name, concepts in CONCEPTS.items():
        current[name] = select_fact(companyfacts, concepts, filing_date,
            duration=name in {'revenue', 'net_income', 'operating_cashflow'})
    revenue = current['revenue']
    prior = select_fact(companyfacts, [revenue['concept']], filing_date, True, revenue['end'])
    ends = {x['end'] for x in current.values()}
    if len(ends) != 1:
        raise ValueError('Financial facts refer to different periods; manually reconcile')
    if not 330 <= (date.fromisoformat(revenue['end'])-date.fromisoformat(prior['end'])).days <= 400:
        raise ValueError('Prior revenue must be from the previous annual period')
    r, old, assets = float(revenue['val']), float(prior['val']), float(current['assets']['val'])
    if r <= 0 or old <= 0 or assets <= 0:
        raise ValueError('Revenue and asset denominators must be positive')
    result = {'sentiment_score': float(sentiment_score), 'revenue_growth': r/old-1,
        'net_margin': float(current['net_income']['val'])/r,
        'cashflow_margin': float(current['operating_cashflow']['val'])/r,
        'liabilities_to_assets': float(current['liabilities']['val'])/assets}
    if not all(math.isfinite(x) for x in result.values()) or not -1 <= result['sentiment_score'] <= 1:
        raise ValueError('Invalid numeric features')
    return {'features': result, 'period_end': revenue['end'],
        'as_of': filing_date, 'facts_used': current, 'prior_revenue': prior}
