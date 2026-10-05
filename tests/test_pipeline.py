import json
import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda
from financial_analysis.sentiment import analyze, validate_output, make_chain
from financial_analysis.features import select_fact, structured_features
from financial_analysis.labels import forward_label
from financial_analysis.demo import generate
from financial_analysis.train import train, validate, split
from app import create_app

TEXT = 'Revenue increased during the year. Supply chain disruptions remain a risk.'

def output():
    return {'summary':'Revenue grew, while supply chain risks remain.', 'sentiment':'positive', 'score':.4,
        'evidence':['Revenue increased during the year.'], 'risks':['Supply chain disruptions']}

def test_fabricated_evidence_and_nan_rejected():
    value = output()
    value['evidence'] = ['Invented record-breaking profit']
    with pytest.raises(ValueError,match='Evidence'):
        validate_output(value,TEXT)
    value = output()
    value['score'] = float('nan')
    with pytest.raises(ValueError,match='finite'):
        validate_output(value,TEXT)

def test_langchain_mocked_analysis_retains_evidence():
    # Mock only: this test does not establish live Mistral output quality.
    result = analyze(TEXT,RunnableLambda(lambda _: output()))
    assert result['sentiment_score'] == .4
    assert result['chunks'][0]['evidence'][0] in TEXT

def facts():
    def record(val,end='2024-12-31',filed='2025-02-01',start=None):
        obj = {'val':val,'end':end,'filed':filed,'form':'10-K'}
        if start: obj['start']=start
        return obj
    values = {'Revenues':[record(100,start='2024-01-01'),record(80,'2023-12-31','2024-02-01','2023-01-01'),record(999,filed='2025-06-01',start='2024-01-01')],
        'NetIncomeLoss':[record(10,start='2024-01-01')], 'Assets':[record(200)],
        'Liabilities':[record(100)], 'NetCashProvidedByUsedInOperatingActivities':[record(20,start='2024-01-01')]}
    return {'facts':{'us-gaap':{key:{'units':{'USD':rows}} for key,rows in values.items()}}}

def test_asof_excludes_future_restatement_and_ratios():
    selected = select_fact(facts(),['Revenues'],'2025-02-01',True)
    assert selected['val'] == 100
    result = structured_features(facts(),'2025-02-01',.4)['features']
    assert result['revenue_growth'] == .25
    assert result['net_margin'] == .1
    assert result['liabilities_to_assets'] == .5

def test_independent_label_starts_after_filing():
    prices = pd.DataFrame({'date':['2025-01-01','2025-01-02','2025-01-03','2025-01-04'], 'adjusted_close':[1000,100,105,110]})
    label = forward_label(prices,'2025-01-01',2)
    assert label['label_start_date'] == '2025-01-02'
    assert label['forward_return'] == pytest.approx(.1)

def test_purge_overlapping_future_return_labels(tmp_path):
    data = validate(pd.read_csv(generate(tmp_path/'demo.csv')))
    train_set,val,test = split(data)
    assert train_set.label_end_date.max() < val.filing_date.min()
    assert val.label_end_date.max() < test.filing_date.min()
    data.loc[0,'source_type']='real_filings'
    with pytest.raises(ValueError,match='never mixed'):
        validate(data)

def test_model_and_dashboard_roundtrip(tmp_path):
    data = generate(tmp_path/'demo.csv')
    report = train(data,tmp_path/'artifacts')
    assert report['source_type'] == 'synthetic_demo'
    client = create_app(tmp_path/'artifacts',RunnableLambda(lambda _:output())).test_client()
    page = client.get('/')
    assert page.status_code == 200 and b'Synthetic demonstration' in page.data
    response = client.post('/predict',json={'sentiment_score':.4,'revenue_growth':.1,
        'net_margin':.1,'cashflow_margin':.2,'liabilities_to_assets':.5})
    assert response.status_code == 200
    assert 0 <= response.json['positive_outlook_probability'] <= 1
    assert client.post('/predict',json=[]).status_code == 400
    assert client.post('/analyze',json={'text':TEXT}).status_code == 200

def test_missing_local_model_has_explicit_error(tmp_path,monkeypatch):
    monkeypatch.delenv('MISTRAL_MODEL_PATH',raising=False)
    client = create_app(tmp_path).test_client()
    assert client.get('/').status_code == 200
    assert client.post('/analyze',json={'text':TEXT}).status_code == 503
    with pytest.raises(ValueError,match='not found'):
        make_chain(tmp_path/'missing.gguf')
