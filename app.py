"""Local dashboard for evidence-backed sentiment and a trained outlook model."""
import json
import math
import os
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request
from xgboost import XGBClassifier
from financial_analysis.features import FEATURES
from financial_analysis.sentiment import analyze, make_chain

ROOT = Path(__file__).resolve().parent

def create_app(artifact_dir=None, chain=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 100_000
    artifacts = Path(artifact_dir or os.environ.get('FINANCIAL_MODEL_DIR', ROOT/'artifacts/demo'))
    report, model, unavailable = None, None, None
    try:
        report = json.loads((artifacts/'evaluation.json').read_text())
        if report['features'] != FEATURES or report['source_type'] not in {'synthetic_demo', 'real_filings'}:
            raise ValueError('Incompatible model metadata')
        model = XGBClassifier()
        model.load_model(artifacts/'model.json')
    except (OSError, ValueError, KeyError) as exc:
        unavailable = f'Model unavailable: {exc}. Run python -m financial_analysis.demo for a synthetic demonstration.'
    chain_holder = [chain]

    @app.get('/')
    def home():
        example_path = ROOT/'artifacts/apple_excerpt_sentiment.json'
        example = json.loads(example_path.read_text()) if example_path.exists() else None
        return render_template('index.html', report=report, error=unavailable,
            example=example,
            llm_configured=chain_holder[0] is not None or bool(os.environ.get('MISTRAL_MODEL_PATH')))

    @app.get('/health')
    def health():
        return jsonify({'classifier_ready': model is not None,
            'source_type': report['source_type'] if report else None,
            'llm_configured': chain_holder[0] is not None or bool(os.environ.get('MISTRAL_MODEL_PATH'))})

    @app.post('/predict')
    def predict():
        if model is None:
            return jsonify({'error': unavailable}), 503
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({'error': 'JSON object required'}), 400
        try:
            values = {key: float(body[key]) for key in FEATURES}
            if any(isinstance(body[key], bool) for key in FEATURES) or not all(math.isfinite(x) for x in values.values()):
                raise ValueError('Features must be finite numbers')
            if not -1 <= values['sentiment_score'] <= 1 or values['liabilities_to_assets'] < 0:
                raise ValueError('Invalid sentiment score or liability ratio')
        except (KeyError, ValueError, TypeError) as exc:
            return jsonify({'error': str(exc)}), 400
        probability = float(model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0,1])
        return jsonify({'positive_outlook_probability': probability,
            'prediction': 'positive' if probability >= .5 else 'negative',
            'source_type': report['source_type'], 'features': values,
            'note': 'Synthetic demonstration only' if report['source_type'] == 'synthetic_demo' else 'Historical research model; probabilities are not calibrated'})

    @app.post('/analyze')
    def sentiment():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get('text'), str) or not 20 <= len(body['text'].strip()) <= 12000:
            return jsonify({'error': 'Provide 20 to 12,000 characters of filing text'}), 400
        if chain_holder[0] is None:
            path = os.environ.get('MISTRAL_MODEL_PATH')
            if not path:
                return jsonify({'error': 'Set MISTRAL_MODEL_PATH and LLAMA_CLI to enable live local Mistral analysis'}), 503
            try:
                chain_holder[0] = make_chain(path, os.environ.get('LLAMA_CLI', 'llama-completion'))
            except (ValueError, OSError) as exc:
                return jsonify({'error': str(exc)}), 503
        try:
            result = analyze(body['text'], chain_holder[0])
        except Exception:
            # Invalid output is rejected; it never silently becomes a neutral score.
            app.logger.exception('Local language-model analysis failed')
            return jsonify({'error': 'Model invocation or evidence validation failed. Inspect the local server log.'}), 502
        return jsonify(result)
    return app

app = create_app()
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5001')), debug=False)
