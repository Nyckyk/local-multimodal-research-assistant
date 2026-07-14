# Local Research Assistant

## Required Ollama models

```powershell
ollama pull qwen3.5:9b
ollama pull qwen3-vl:8b-instruct
ollama create research-vision:latest -f Modelfile.vision
```

Confirm the custom vision model:

```powershell
ollama show research-vision:latest --modelfile
```

It should include `PARAMETER num_ctx 8192`.

## Install Python packages

```powershell
pip install -r requirements.txt
```

## Test vision independently

Run this from the project root:

```powershell
python scripts/test_vision.py
```

## Start the app

```powershell
streamlit run app.py
```

For figure or table questions, enable **Use vision for this question**, select the PDF and enter the PDF page number.

The vision service automatically retries a page at lower image resolutions when Ollama reports that the image exceeds its context window. Vision failures are shown inside the debug panel instead of crashing the Streamlit app.

## Automated regression suite

The suite has fast mocked-model tests, fixture-driven real local-Ollama tests,
and one optional Streamlit/Playwright smoke test. It checks the exact text and
vision model names configured in `settings.py` before real-model work and writes
timestamped reports under `test_results/`.

```powershell
python scripts/run_regression_suite.py --fast
python scripts/run_regression_suite.py --ollama
python scripts/run_regression_suite.py --all
python scripts/run_regression_suite.py --all --smoke  # optional
```

Each run saves `summary.md`, `results.json`, raw model responses, validated
structured outputs, and failure details. Scientific expected values in
`tests/fixtures/regression_cases.yaml` are protected: Codex must not alter them
merely to obtain passing tests, and changing those values requires explicit user
approval. Production code must not import test fixtures.
