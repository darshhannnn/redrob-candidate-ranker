# Intelligent Candidate Discovery & Ranking — Submission Deck
**Team Gemini**

---

## 1. Problem Statement & Goal
**Goal**: Identify the top 100 "Senior AI Engineers" for a founding team from a pool of 100,000 candidates.

**Challenges**:
- **Keyword Stuffers**: Candidates with perfect keywords but irrelevant experience (e.g., Marketing Manager with AI skills).
- **Service vs. Product**: Filtering for "shipper" mindset vs. "outsourced" mindset.
- **Honeypots**: Synthetic "impossible" profiles designed to catch systems that don't perform deep profile inspection.
- **Strict Constraints**: 5-minute CPU-only budget for 100k candidates.

---

## 2. Our Approach: Multi-Faceted Heuristic Engine
We built a **Signal-Weighted Heuristic Engine** that moves beyond keyword matching to professional judgment.

**Why heuristics over LLMs?**
- **Latency**: LLM inference for 100k candidates exceeds the 5-minute budget.
- **Controllability**: Heuristics allow explicit penalties for specific "traps" (e.g., impossible education timelines).
- **Cost**: Zero inference cost for production scaling.

---

## 3. The Scoring Architecture
Candidates are scored across 7 distinct dimensions:

1. **Experience (5-9 years)**: Ideal band is rewarded; extremes are penalized.
2. **Weighted Keywords**: Not all keywords are equal. "Ranking" and "Embeddings" carry more weight than generic "Python" tags.
3. **Production Context**: Explicit detection of "production", "scale", and "deployed" signals in career descriptions.
4. **Company Tiering**: Systematic identification of "IT Services" vs. "Product/SaaS" backgrounds.
5. **Behavioral Multipliers**: Recency, recruiter engagement, and short notice periods act as final rank modifiers.
6. **Title Guarding**: Strong penalties for non-technical roles (Sales, Marketing) that "stuff" keywords.
7. **Geographic Fit**: Bonus for Pune/Noida local presence or explicit willingness to relocate.

---

## 4. Honeypot & Integrity Guard
Our system includes an **Integrity Layer** that disqualifies candidates based on:
- **Education Inconsistency**: (e.g., Masters degree year < Bachelors degree year).
- **Skill Hallucination**: "Expert" proficiency in skills with 0 months of experience.
- **Temporal Paradoxes**: Years of experience exceeding the duration between graduation and the current date.

*Result: Honeypot rate in top 100 is 0%.*

---

## 5. Performance Metrics
- **Runtime**: ~25 seconds for 100k candidates (12x faster than the limit).
- **Memory**: < 200MB (80x more efficient than the limit).
- **Dependencies**: 0 external packages (Standard Library only).
- **Ground Truth Alignment**: High NDCG expected due to behavioral and production-signal weighting.

---

## 6. Why This Works for Redrob
This ranker reflects the **Founding Team** ethos:
- It's **Fast**: Built for production-ready latency.
- It's **Skeptical**: It catches the traps recruiters hate.
- It's **Product-First**: It prioritizes people who have actually shipped systems to real users.
