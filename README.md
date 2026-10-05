# LLM-Based Financial Analysis Lab

A research prototype that connects SEC filing text, **local Mistral-7B sentiment analysis through LangChain**, **as-of structured financial ratios**, and an **XGBoost future-return-direction classifier**, with a local Flask dashboard.

The original 2025 code retrieved and summarized a filing. The sentiment, feature integration, classifier, dashboard, and test workflow below were added in **October 2026**. They should not be described as completed in July–September 2025.

## What works and what has been verified

| Component | Status |
| --- | --- |
| Dashboard and XGBoost train/export/predict | Tested locally with the checked-in synthetic demonstration |
| As-of SEC fact selection and ratios | Tested with controlled fact fixtures, including future restatement rejection |
| Independent return labels and split purging | Tested with controlled price fixtures |
| LangChain prompt, local llama-completion invocation, JSON and exact-quote validation | Tested with an actual local Mistral-7B-Instruct v0.2 Q4_K_M inference on one Apple filing excerpt |
| Live Mistral generation | One 3,000-character Apple 2024 excerpt was run successfully; the saved output and exact source are included |
| Real-market classifier accuracy | Not established; no real labeled multi-filing cohort is supplied |

The verified example is `data/apple_excerpt_2024.txt` and `artifacts/apple_excerpt_sentiment.json`. It returned neutral disclosure sentiment, a summary, risks, and three exact quotes that passed source validation. The output records the model hash and llama.cpp runner version. This verifies one integration run, not sentiment accuracy or full-filing coverage.

**Synthetic scores demonstrate the software workflow, not real financial forecasting performance.** No sentiment output is invented when a local model is absent.

## Run the dashboard

Python 3.11+ recommended. From the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5001. The included `artifacts/demo` model makes the classifier immediately usable. The page prominently identifies synthetic training. Regenerate the same seeded 240-row example with:

```bash
python -m financial_analysis.demo
```

## Enable real local Mistral analysis

Install llama.cpp (build the `llama-completion` target) and obtain a compatible Mistral-7B-Instruct GGUF model separately. Large model weights are not committed.

```bash
python -m financial_analysis.sentiment --input data/apple_mda_2024.txt --output artifacts/apple_sentiment.json --model /absolute/path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf --llama-cli /absolute/path/to/llama-completion
```

Older llama.cpp releases may call this runner `llama-cli`; the program checks for the required completion-mode flags. Newer interactive `llama-cli` builds are not interchangeable with `llama-completion`.

For dashboard analysis, set `MISTRAL_MODEL_PATH` and optionally `LLAMA_CLI` before starting the app. The model receives short filing chunks and returns a concise summary, positive/neutral/negative business-condition sentiment, a score, disclosed risks, and exact supporting quotes. Output with invalid scores, inconsistent labels, or invented quotes is rejected. Chunk scores are averaged by character count; this is not a calibrated sentiment measure. Exact quote checking does not independently establish that the summary or sentiment interpretation is correct.

## Structured financial inputs

The `financial_analysis.features` module reads SEC Company Facts JSON and selects annual USD facts whose **filing date is no later than the analysis date**. Future restatements are excluded. It checks annual duration and matching period ends, then calculates:

- annual revenue growth;
- net income / revenue;
- operating cash flow / revenue;
- liabilities / assets.

Missing or conflicting facts produce an error rather than guessed values. Custom taxonomies, non-USD reporting, and unsupported company-specific concept mappings require manual reconciliation.

Download Company Facts or submissions after setting `SEC_USER_AGENT` to your name and real contact email:

```bash
python -m financial_analysis.sec --cik 320193 --kind companyfacts --out data/apple_companyfacts.json
```

SEC API documentation: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

## Train on a real historical cohort

Supply a CSV manifest with these columns (paths relative to the manifest):

```text
ticker,filing_date,filing_text,companyfacts_json,sentiment_json,prices_csv,price_source
```

For each filing, provide its actual text, downloaded Company Facts, validated local-Mistral sentiment JSON, and independently sourced daily prices with columns `date,adjusted_close`. Document the price source. Sentiment hashes must match the source text.

```bash
python -m financial_analysis.prepare --manifest data/manifest.csv --out data/training.csv
python -m financial_analysis.train --data data/training.csv --out artifacts/real
```

Set `FINANCIAL_MODEL_DIR` to `artifacts/real` to use the real-data model in the dashboard.

The target is positive when the adjusted return is greater than zero, negative otherwise. Entry is the closing price of the first trading session after the filing date, and exit is 20 price observations later. Inputs should contain one row per trading session. This avoids using a pre-disclosure entry price, but it is a simplified research target: transaction costs, availability delays, benchmark-relative returns, and survivorship bias are not modeled.

Training uses a chronological 60/20/20 date split. Rows whose future-return labels extend into the next split are purged. Fixed parameters are used; test results are not used for tuning. Synthetic and real rows cannot be mixed. Reports include dataset hash, date ranges, baseline comparisons, Brier score, ROC AUC, and feature importance. A small same-company cohort is insufficient to establish generalization.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover fabricated evidence, invalid scores, as-of fact selection, independent labels, overlapping-label purging, model export, dashboard requests, and absent-model behavior. Mocked LLM tests do not establish live Mistral quality.

## Source layout

- `financial_analysis/sentiment.py`: LangChain and local Mistral integration
- `features.py`, `sec.py`: as-of ratios and SEC retrieval
- `labels.py`, `prepare.py`: independently derived labels and provenance joins
- `train.py`, `demo.py`: classifier and clearly separated synthetic example
- `app.py`, `templates/index.html`: interactive local dashboard
- `code/`: original preparation scripts; `query_llm.py` now forwards to the portable CLI
- `artifacts/apple_excerpt_sentiment.json`: actual local Mistral output with validated evidence and model provenance
- `data/apple_*`: original filing text, extracted section, and the tested excerpt. The original empty summary file is not evidence of successful LLM execution.

Developed with AI assistance. This is a local research application, not a deployed investment product.
