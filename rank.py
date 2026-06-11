import json
import csv
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Optional imports for semantic search and LLM
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False

try:
    import ollama
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Configuration
TOP_N = 100
SEMANTIC_MODEL_NAME = 'all-mpnet-base-v2'
LLM_MODEL_NAME = 'qwen2.5:0.5b' 
CHROMA_PATH = "./chroma_db"

# Keywords and Constants
NON_TECH_TITLES = {"marketing", "sales", "hr", "human resources", "recruiter", "accountant", "graphic designer", "content writer", "operations manager"}
AI_KEYWORDS = [
    "embedding", "retrieval", "vector database", "ranking", "llm",
    "nlp", "rag", "transformer", "bert", "gpt", "fine-tuning",
    "recommendation system", "personalization", "search relevance",
    "semantic search", "faiss", "pinecone", "langchain", "llamaindex",
    "reranking", "ndcg", "mrr", "evaluation framework", "mlops"
]

SERVICE_SIGNALS = ["consultant", "associate consultant", "delivery manager", "client engagement", "offshore", "onsite coordinator", "outsourcing"]
PRODUCT_SIGNALS = ["founding engineer", "staff engineer", "principal engineer", "growth", "0 to 1", "built from scratch", "product", "platform", "saas"]

PRODUCTION_VERBS_A = ["deployed", "shipped", "production", "latency", "scale", "serving", "real-time", "millions"]
PRODUCTION_VERBS_B = ["built", "optimized", "improved", "reduced", "system", "pipeline"]

def check_honeypot(candidate):
    # 1. Expert with 0 months
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            return True
            
    # 2. Years of experience vs career history
    total_months = sum(job.get('duration_months', 0) for job in candidate.get('career_history', []))
    claimed_years = candidate.get('profile', {}).get('years_of_experience', 0)
    if total_months / 12 > claimed_years + 3: 
        return True
    
    # 3. Impossible experience
    if claimed_years > 45: return True
    
    # 4. Education sequence check
    edu_list = candidate.get('education', [])
    if len(edu_list) > 1:
        b_year = None
        m_year = None
        for edu in edu_list:
            deg = edu.get('degree', '').lower()
            year = edu.get('start_year')
            if not year: continue
            if 'b.tech' in deg or 'b.e.' in deg or 'bachelor' in deg:
                if b_year is None or year < b_year: b_year = year
            if 'm.tech' in deg or 'm.e.' in deg or 'master' in deg or 'ph.d' in deg or 'phd' in deg:
                if m_year is None or year < m_year: m_year = year
        if b_year and m_year and m_year < b_year:
            return True

    # 5. Education vs experience timeline
    first_edu_year = 2030
    for edu in edu_list:
        start = edu.get('start_year')
        if start and start < first_edu_year:
            first_edu_year = start
    current_year = 2026
    if current_year - first_edu_year < claimed_years - 2:
        return True

    # 6. Aggressive Title-stuffing check
    current_title = candidate.get('profile', {}).get('current_title', '').lower()
    if any(t in current_title for t in NON_TECH_TITLES):
        summary = candidate.get('profile', {}).get('summary', '').lower()
        skills = " ".join([s['name'].lower() for s in candidate.get('skills', [])])
        match_count = sum(1 for kw in AI_KEYWORDS if kw in summary or kw in skills)
        if match_count >= 8: 
            return True
        if "graphic" in current_title or "sales" in current_title:
             if match_count >= 5: return True

    return False

def get_scores(candidate, semantic_score=0.0):
    profile = candidate['profile']
    career_history = candidate.get('career_history', [])
    career_text = " ".join([job.get('description', '').lower() for job in career_history])
    all_history_text = " ".join([f"{job.get('title', '')} {job.get('description', '')}".lower() for job in career_history])
    
    # --- A. Feature Score (0-100 base) ---
    feat_score = 0
    
    # 1. Experience depth (0-20)
    yoe = profile.get('years_of_experience', 0)
    if 5 <= yoe <= 9: feat_score += 20
    elif 4 <= yoe <= 12: feat_score += 14
    else: feat_score += 5
    if yoe > 15: feat_score -= 5 

    # 2. Production signal (0-20)
    prod_score = 0
    for verb in PRODUCTION_VERBS_A:
        if verb in career_text: prod_score += 4
    for verb in PRODUCTION_VERBS_B:
        if verb in career_text: prod_score += 2
    feat_score += min(prod_score, 20)

    # 3. Semantic match (0-20)
    feat_score += min(semantic_score * 20, 20)
    
    # 4. Company Type (0-20) - Title + Description Heuristic
    company_score = 0
    if any(sig in all_history_text for sig in PRODUCT_SIGNALS):
        company_score += 15
        if "founding" in all_history_text or "0 to 1" in all_history_text:
            company_score += 5
    if any(sig in all_history_text for sig in SERVICE_SIGNALS):
        company_score -= 10
    feat_score += max(0, min(company_score, 20))

    # 5. Education alignment (0-20)
    edu_score = 0
    for edu in candidate.get('education', []):
        fos = edu.get('field_of_study', '').lower()
        if any(kw in fos for kw in ["computer science", "artificial intelligence", "machine learning"]):
            edu_score = 20
            break
        elif "engineering" in fos:
            edu_score = 14
    feat_score += edu_score

    # --- B. Behavioral Score (0-100 normalized) ---
    behav_score = 0
    signals = candidate.get('redrob_signals', {})
    
    last_active = signals.get('last_active_date', '2000-01-01')
    try:
        dt = datetime.strptime(last_active, '%Y-%m-%d')
        days_since = (datetime(2026, 6, 11) - dt).days
        if days_since < 30: behav_score += 40
        elif days_since < 90: behav_score += 20
    except: pass
    
    if signals.get('notice_period_days', 90) < 30: behav_score += 40
    elif signals.get('notice_period_days', 90) <= 60: behav_score += 20
    
    if signals.get('recruiter_response_rate', 0) > 0.7: behav_score += 20
    elif signals.get('recruiter_response_rate', 0) > 0.4: behav_score += 10
    
    return feat_score, behav_score

def get_llm_score(candidate, job_desc):
    if not HAS_LLM:
        return 0, "Detailed analysis unavailable.", "N/A", "N/A"
    
    profile_summary = f"""
    Title: {candidate['profile']['current_title']}
    Experience: {candidate['profile']['years_of_experience']} years
    History: {'; '.join([job.get('title', '') + ' at ' + job.get('company', '') for job in candidate.get('career_history', [])[:3]])}
    Summary: {candidate['profile']['summary'][:300]}
    """
    
    prompt = f"""You are a senior technical recruiter for a founding AI/ML engineering role.
JD (Scale requirement: high-traffic, millions of queries):
{job_desc[:500]}

CANDIDATE:
{profile_summary}

INSTRUCTIONS:
1. Scale Weighting: Prioritize candidates who built systems at high-traffic scale (millions of queries/day). Verifiable company context (e.g. Amazon, Flipkart) ranks higher than smaller equivalents.
2. Skepticism: Treat self-reported metrics (e.g. "0.91 NDCG", "30% revenue lift") with moderate skepticism unless backed by company scale or team context.
3. Mandatory Gap: You MUST provide a top_gap for every candidate. If technical skills are elite, focus on startup-fit gaps (e.g. "Adapting from large-corp infra to 0-to-1 environments") or compensation/relocation risks. Never return "None".

Return ONLY valid JSON:
{{
  "llm_score": <integer 0-100>,
  "fit_summary": "<one sentence fit justification>",
  "top_strength": "<single strongest signal>",
  "top_gap": "<single biggest concern>"
}}"""

    try:
        response = ollama.chat(model=LLM_MODEL_NAME, messages=[{'role': 'user', 'content': prompt}], format='json')
        data = json.loads(response['message']['content'])
        return data.get('llm_score', 50), data.get('fit_summary', ''), data.get('top_strength', ''), data.get('top_gap', '')
    except Exception:
        prof = candidate['profile']
        summary = f"Strong candidate with {prof['years_of_experience']}y experience in {prof['current_industry']}."
        strength = f"Experience with {candidate['skills'][0]['name']}" if candidate['skills'] else "Solid background"
        gap = "Requires deeper verification of production scale."
        return 0, summary, strength, gap

def main():
    parser = argparse.ArgumentParser(description='Hybrid Semantic + LLM Candidate Ranker')
    parser.add_argument('--candidates', type=str, required=True, help='Path to candidates.jsonl')
    parser.add_argument('--out', type=str, default='1.csv', help='Output path')
    parser.add_argument('--jd', type=str, default='job_description.txt', help='Path to JD text')
    args = parser.parse_args()

    if not os.path.exists(args.jd):
        print(f"Error: JD file {args.jd} not found.")
        return

    with open(args.jd, 'r', encoding='utf-8') as f:
        job_description = f.read()

    print("Stage 0: Integrity Guard...")
    filtered_candidates = []
    with open(args.candidates, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            if not check_honeypot(cand):
                filtered_candidates.append(cand)
    print(f"Passed integrity check: {len(filtered_candidates)}")

    semantic_scores = {}
    if HAS_SEMANTIC:
        print("Stage 1: Semantic Retrieval...")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(name="candidates")
        
        if collection.count() == 0:
            print("Embedding candidates (this may take a while on CPU)...")
            model = SentenceTransformer(SEMANTIC_MODEL_NAME)
            batch_size = 128
            for i in tqdm(range(0, len(filtered_candidates), batch_size)):
                batch = filtered_candidates[i:i+batch_size]
                texts = [f"{c['profile']['current_title']} {c['profile']['summary']} {' '.join([s['name'] for s in c['skills']])}" for c in batch]
                ids = [c['candidate_id'] for c in batch]
                embeddings = model.encode(texts).tolist()
                collection.add(ids=ids, embeddings=embeddings)
        
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        jd_embedding = model.encode([job_description])[0].tolist()
        results = collection.query(query_embeddings=[jd_embedding], n_results=min(len(filtered_candidates), 80))
        for cid, dist in zip(results['ids'][0], results['distances'][0]):
            semantic_scores[cid] = 1.0 - (dist / 2.0)
    else:
        for c in filtered_candidates[:80]:
            semantic_scores[c['candidate_id']] = 0.5

    print("Stage 2: Scoring...")
    candidate_dict = {c['candidate_id']: c for c in filtered_candidates}
    shortlisted_data = []
    for cid, sem_score in semantic_scores.items():
        if cid not in candidate_dict: continue
        cand = candidate_dict[cid]
        feat_score, behav_score = get_scores(cand, sem_score)
        shortlisted_data.append({
            'candidate_id': cid, 
            'feat_score': feat_score, 
            'behav_score': behav_score,
            'sem_score': sem_score
        })
    
    # Sort by feature score for LLM shortlist
    shortlisted_data.sort(key=lambda x: (-x['feat_score'], x['candidate_id']))
    top_20 = shortlisted_data[:20]
    
    print("Stage 3: LLM Reranking...")
    final_results = []
    for entry in tqdm(top_20):
        cand = candidate_dict[entry['candidate_id']]
        llm_score, fit_summary, strength, gap = get_llm_score(cand, job_description)
        
        # Balanced formula: 45% LLM + 30% Features + 25% Behavioral
        final_score = (0.45 * llm_score) + (0.30 * entry['feat_score']) + (0.25 * entry['behav_score'])
        
        final_results.append({
            'candidate_id': cand['candidate_id'],
            'name': cand['profile'].get('anonymized_name', 'Unknown'),
            'final_score': round(final_score / 100.0, 3),
            'llm_score': llm_score,
            'feat_score': entry['feat_score'],
            'top_strength': strength,
            'top_gap': gap,
            'fit_summary': fit_summary
        })

    # Fill rest to 100
    if len(final_results) < TOP_N:
        for entry in shortlisted_data[20:TOP_N]:
            cand = candidate_dict[entry['candidate_id']]
            # For these, llm_score is 0
            final_score = (0.30 * entry['feat_score']) + (0.25 * entry['behav_score'])
            final_results.append({
                'candidate_id': cand['candidate_id'],
                'name': cand['profile'].get('anonymized_name', 'Unknown'),
                'final_score': round(final_score / 100.0, 3),
                'llm_score': 0,
                'feat_score': entry['feat_score'],
                'top_strength': "Strong technical profile",
                'top_gap': "Detailed LLM analysis pending",
                'fit_summary': "Qualified candidate from semantic search."
            })

    final_results.sort(key=lambda x: (-x['final_score'], x['candidate_id']))
    
    print(f"Writing output to {args.out}...")
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'name', 'final_score', 'llm_score', 'feature_score', 'top_strength', 'top_gap', 'fit_summary'])
        for i, res in enumerate(final_results[:TOP_N]):
            writer.writerow([
                res['candidate_id'], i + 1, res['name'], res['final_score'], 
                res['llm_score'], res['feat_score'], res['top_strength'], 
                res['top_gap'], res['fit_summary']
            ])

if __name__ == "__main__":
    main()
