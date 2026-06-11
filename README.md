# Intelligent Candidate Discovery & Ranking — Hybrid Ranker

This repository contains the source code for the Redrob Hackathon "Intelligent Candidate Discovery & Ranking Challenge".

## System Overview: Hybrid Semantic + LLM Pipeline
Our system implements a sophisticated two-stage hybrid pipeline designed to identify high-potential "Senior AI Engineers" while filtering out synthetic traps and keyword stuffers.

### Architecture:
1.  **Stage 0: Integrity Guard**: Hard-disqualifies "honeypot" candidates with impossible education timelines, zero-duration expertise, or non-technical title stuffing.
2.  **Stage 1: Semantic Retrieval**: Uses `all-mpnet-base-v2` and ChromaDB to perform vector-based recall across the 100k candidate pool, selecting the top 80 most semantically relevant profiles.
3.  **Stage 2: Signal-Weighted Feature Scoring**: Re-ranks the shortlist using non-linear experience curves, production signal detection (verbs like "deployed", "scaled"), and behavioral signals (recency, notice period).
4.  **Stage 3: LLM Reranking (Precision Layer)**: The top 20 candidates are analyzed by a local LLM (Ollama) for deep contextual fit, generating human-readable reasoning and strengths/gaps analysis.

## Setup & Reproduction

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com/) (running with `qwen2.5:0.5b` or similar)
- `pip install -r requirements.txt`

### Step 1: Pre-compute Embeddings (allowed to exceed 5 mins)
Before ranking, populate the local vector database:
```bash
python precompute_embeddings.py --candidates ./data/candidates.jsonl
```

### Step 2: Generate Ranked Output (must be < 5 mins)
Execute the hybrid ranker to produce the final CSV:
```bash
python rank.py --candidates ./data/candidates.jsonl --out ./1.csv --jd ./job_description.txt
```

## Repository Structure
- `rank.py`: The hybrid ranking and reranking engine.
- `precompute_embeddings.py`: Script to populate the ChromaDB vector store.
- `1.csv`: The validated output file.
- `submission_metadata.yaml`: Team and environment metadata.
- `approach_deck.md`: Detailed explanation of the methodology.
- `chroma_db/`: Persistent vector storage directory.

## Technical Constraints Compliance
- **Runtime**: < 1 minute (with pre-computed index).
- **Compute**: CPU-only, no network calls during Stage 2/3.
- **Explainability**: Every top-tier candidate includes an LLM-generated reasoning string.
