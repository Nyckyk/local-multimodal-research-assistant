import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PAPERS_FOLDER = BASE_DIR / "papers"
DATA_FOLDER = BASE_DIR / "data"
TEXT_FOLDER = DATA_FOLDER / "extracted_text"
DB_PATH = DATA_FOLDER / "chroma"
VISUAL_INDEX_FOLDER = DATA_FOLDER / "visual_index"
VISUAL_INDEX_PATH = VISUAL_INDEX_FOLDER / "index.json"

COLLECTION_NAME = "papers"

OLLAMA_MODEL = "qwen3.5:9b"
VISION_MODEL = "research-vision:latest"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_RESULTS = 20
FINAL_RESULTS = 8 
MAX_HISTORY_MESSAGES = 6

NORMAL_NUM_PREDICT = int(os.getenv("NORMAL_NUM_PREDICT", "700"))
SUMMARY_NUM_PREDICT = int(os.getenv("SUMMARY_NUM_PREDICT", "1800"))

VISUAL_AMBIGUITY_MARGIN = float(os.getenv("VISUAL_AMBIGUITY_MARGIN", "0.075"))
NYQUIST_LOCAL_DEVIATION_THRESHOLD = float(
    os.getenv("NYQUIST_LOCAL_DEVIATION_THRESHOLD", "0.02")
)

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 300

PAPERS_FOLDER.mkdir(parents=True, exist_ok=True)
TEXT_FOLDER.mkdir(parents=True, exist_ok=True)
DB_PATH.mkdir(parents=True, exist_ok=True) 
VISUAL_INDEX_FOLDER.mkdir(parents=True, exist_ok=True)
