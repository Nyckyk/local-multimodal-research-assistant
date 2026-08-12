"""Run real-model end-to-end scientific RAG evaluation suites."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The application is intentionally local-only. Prevent the model loaders from
# probing Hugging Face during evaluation; configured models must already exist.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.runner import (
    default_output_directory,
    execute_run,
    load_suite,
    write_artifacts,
)
from rag.database import get_collection
from rag.embeddings import load_embedder, load_reranker
from scripts.regression_support import ModelAvailabilityError, installed_ollama_models
from services.research_assistant import ResearchAssistant
from services.visual_index import load_or_build_visual_index
from settings import OLLAMA_MODEL, PAPERS_FOLDER, VISION_MODEL


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="JSON evaluation fixture")
    parser.add_argument("--runs", type=int, default=1, help="number of independent stability runs")
    parser.add_argument("--verbose", action="store_true", help="print each answer after scoring")
    parser.add_argument("--output-dir", type=Path, help="new artifact directory; must not already exist")
    parser.add_argument("--question", help="run one test id (plus declared conversation setup)")
    parser.add_argument("--no-vision", action="store_true", help="disable vision calls while keeping real text RAG")
    return parser.parse_args()


def _metadata_sources(collection) -> set[str]:
    values = collection.get(include=["metadatas"])
    names = set()
    for metadata in values.get("metadatas", []) if isinstance(values, dict) else []:
        if not isinstance(metadata, dict):
            continue
        value = metadata.get("pdf", metadata.get("source"))
        if value:
            names.add(Path(str(value)).name.casefold())
    return names


def preflight(suite: dict, *, no_vision: bool = False) -> tuple[dict, object, dict]:
    """Validate local infrastructure without converting errors to scientific FAIL."""
    models = installed_ollama_models()
    required_models = [OLLAMA_MODEL] + ([] if no_vision else [VISION_MODEL])
    missing_models = [model for model in required_models if model not in models]
    if missing_models:
        raise ModelAvailabilityError(
            "Missing configured Ollama model(s): " + ", ".join(missing_models)
        )
    collection = get_collection()
    total = collection.count()
    if total <= 0:
        raise RuntimeError("Chroma collection contains no indexed chunks.")
    document = suite.get("document")
    if document:
        if not (PAPERS_FOLDER / document).exists():
            raise RuntimeError(f"Expected PDF is absent from papers/: {document}")
        indexed_stems = {Path(name).stem.casefold() for name in _metadata_sources(collection)}
        if Path(document).stem.casefold() not in indexed_stems:
            raise RuntimeError(f"Expected paper is not indexed in Chroma: {document}")
    visual_index = load_or_build_visual_index()
    diagnostics = {
        "status": "PASS",
        "ollama_models": ", ".join(sorted(models)),
        "vision": "disabled by --no-vision" if no_vision else "configured local model available",
        "collection_chunks": total,
        "expected_paper_indexed": bool(document),
    }
    return diagnostics, collection, visual_index


def main() -> int:
    args = _arguments()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    suite_path = args.suite if args.suite.is_absolute() else ROOT / args.suite
    suite = load_suite(suite_path)
    output = args.output_dir or default_output_directory(ROOT, suite["name"])
    if not output.is_absolute():
        output = ROOT / output
    # A timestamp has one-second resolution. Avoid overwriting when a short
    # probe is invoked twice within that same second.
    if args.output_dir is None and output.exists():
        suffix = 2
        candidate = Path(f"{output}_{suffix}")
        while candidate.exists():
            suffix += 1
            candidate = Path(f"{output}_{suffix}")
        output = candidate
    try:
        diagnostics, collection, visual_index = preflight(suite, no_vision=args.no_vision)
        backend = ResearchAssistant(
            collection=collection,
            embedder=load_embedder(),
            reranker=load_reranker(),
            visual_index=visual_index,
        )
    except Exception as error:
        diagnostics = {"status": "ERROR", "detail": f"{type(error).__name__}: {error}"}
        write_artifacts(output, suite, [], diagnostics)
        print(f"ERROR — {diagnostics['detail']}")
        print(f"Report: {output / 'report.md'}")
        return 2

    def show(result):
        print(f"{result['id']} {result['status']} ({result.get('score', 0) * 100:.1f}%)")
        if args.verbose:
            print(result.get("answer") or result.get("error") or "")
            print()

    runs = [
        execute_run(
            suite, backend, run_number=number, question_id=args.question,
            allow_vision=not args.no_vision, on_result=show,
        )
        for number in range(1, args.runs + 1)
    ]
    scores = write_artifacts(output, suite, runs, diagnostics)
    aggregate = scores["aggregate"]
    counts = aggregate["counts"]
    print(
        f"PASS={counts.get('PASS', 0)} PARTIAL={counts.get('PARTIAL', 0)} "
        f"FAIL={counts.get('FAIL', 0)} ERROR={counts.get('ERROR', 0)} "
        f"pass_rate={aggregate['pass_rate'] * 100:.1f}%"
    )
    print(f"Report: {output / 'report.md'}")
    print(f"Answers: {output / 'answers.json'}")
    return 2 if counts.get("ERROR") else 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
