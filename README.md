@'
# Local Multimodal Research Assistant

A fully local AI research assistant for querying scientific papers, retrieving evidence, and analysing complex figures and tables.

The system combines retrieval augmented generation with multimodal document analysis using Ollama, ChromaDB, sentence transformers, PyMuPDF, reranking, and a local Qwen vision model. It is designed to keep document processing and inference local while producing evidence grounded answers from scientific PDFs.

## Key Features

- Local retrieval augmented generation over scientific PDFs
- Semantic search with sentence transformer embeddings
- Persistent vector storage with ChromaDB
- Retrieval reranking before answer generation
- Automatic figure and table reference detection
- Multimodal analysis of scientific figures using a local vision model
- Structured JSON extraction and validation for complex diagrams
- Overlapping regional crops for dense multi section figures
- Automatic retry and graceful fallback when structured visual analysis fails
- Follow-up support for references such as `Figure 9`, `Fig. 13b`, and `What about panel c?`
- Evidence aware answer composition
- Automated regression testing using mocked and real local models
- End to end scientific RAG evaluation
- Fully local inference through Ollama

## Architecture

```text
Scientific PDFs
      |
      v
   PyMuPDF
      |
      +-------------------------+
      |                         |
      v                         v
Text extraction           Figure / table detection
      |                         |
      v                         v
Sentence Transformers      Visual target resolver
      |                         |
      v                         v
   ChromaDB              Page / region rendering
      |                         |
      v                         v
Retrieval + reranking      Qwen vision model
      |                         |
      |                  Structured evidence
      |                         |
      +------------+------------+
                   |
                   v
          Evidence validation
                   |
                   v
            Local Qwen LLM
                   |
                   v
          Grounded final answer
                   |
                   v
          Streamlit interface