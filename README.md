# Intelligent Candidate Discovery & Ranking Ranker

This repository contains the source code for the Redrob Hackathon "Intelligent Candidate Discovery & Ranking Challenge".

## System Overview
The ranking system uses a multi-layered heuristic scoring engine designed to identify "Senior AI Engineers" who possess not only the technical skills but also the "shipper" mindset and product-company background required for a founding team role.

### Key Features:
- **Heuristic Scoring Engine**: Evaluates candidates across 7 dimensions: Experience, Technical Keywords (weighted), Production Experience, Product vs Service background, Title relevance, Location, and Behavioral signals.
- **Honeypot Protection**: Built-in detection for "impossible" profiles, including expertise with 0 months of use, impossible education timelines, and unrealistic experience durations.
- **Behavioral Signal Weighting**: Integrates 23 simulated platform signals (recency, engagement, notice period) to prioritize candidates who are available and responsive.
- **Deterministic Tie-Breaking**: Ensures consistent ranking for candidates with identical scores by using `candidate_id` as a secondary sort key.
- **High Efficiency**: Processes 100,000 candidates in ~25 seconds on a standard CPU, using zero external dependencies (standard library only).

## Setup & Reproduction

### Prerequisites
- Python 3.8+
- Standard library only (no external packages required for `rank.py`)

### Execution
To reproduce the submission CSV:

```bash
python rank.py --candidates ./candidates.jsonl --out ./1.csv
```

The script expects the `candidates.jsonl` file as input and will output exactly 100 candidates in the required CSV format.

## Repository Structure
- `rank.py`: The core ranking and scoring engine.
- `1.csv`: The validated output file.
- `submission_metadata.yaml`: Team and environment metadata.
- `approach_deck.md`: Detailed explanation of the methodology (content for the submission deck).

## Technical Constraints Compliance
- **Runtime**: < 30 seconds for 100k candidates.
- **Memory**: < 200MB (well within the 16GB limit).
- **Environment**: CPU-only, zero network calls, standard library only.
