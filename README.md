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

Figure and table questions are detected automatically. The app builds a local,
incremental visual index from captions and nearby PDF text, resolves references
such as `Figure 9`, `Fig. 13b`, `Table B.3`, and follow-up questions such as
`What about panel c?`, then sends only the resolved page to the vision model.
The index is stored under `data/visual_index/` and is rebuilt only when PDFs are
added, removed, or changed.

Use **Preferred PDF** when duplicate figure numbers need a source hint. If the
reference is still ambiguous, the app asks you to choose a candidate. The
**Manual visual target override** remains available for scanned documents or
unusual numbering: select the PDF and enter the 1-based PDF page number.

Automatic detection can be disabled per question. Text-only questions continue
through the normal RAG path without invoking the vision model.

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

# Automated scientific RAG evaluation

The end-to-end evaluator calls the same `ResearchAssistant.ask(...)` backend as
Streamlit. It uses the configured local Chroma collection, embedding/reranking
models, Ollama text model, and (when applicable) Ollama vision model.

```powershell
python scripts/run_e2e_rag_eval.py --suite tests/evals/senescence_v13.json
python scripts/run_e2e_rag_eval.py --suite tests/evals/senescence_v13.json --question Q05
python scripts/run_e2e_rag_eval.py --suite tests/evals/senescence_v13.json --runs 3
```

Each run writes `answers.json`, `answers.md`, `scores.json`, `report.md`, and
per-question debug JSON beneath `results/e2e_eval/<suite>/<timestamp>/`.
