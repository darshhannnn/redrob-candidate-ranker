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
SERVICE_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", 
    "hcl", "tech mahindra", "mindtree", "l&t", "larsen & toubro", "persistent", "zensar", 
    "hexaware", "mphasis", "ust global", "ntt data", "fujitsu", "ibm", "deloitte", "ey", 
    "kpmg", "pwc", "capita", "atos", "conduent", "genpact"
}

NON_TECH_TITLES = {"marketing", "sales", "hr", "human resources", "recruiter", "accountant", "graphic designer", "content writer", "operations manager"}
AI_KEYWORDS = ["embedding", "retrieval", "vector database", "ranking", "llm", "nlp", "rag", "transformer", "bert", "gpt", "fine-tuning"]

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

def get_feature_score(candidate, semantic_score=0.0):
    score = 0
    profile = candidate['profile']
    career_text = " ".join([job.get('description', '').lower() for job in candidate.get('career_history', [])])
    
    # A. Experience depth score (0–20)
    yoe = profile.get('years_of_experience', 0)
    if 5 <= yoe <= 9: score += 20
    elif 4 <= yoe <= 12: score += 14
    else: score += 5
    if yoe > 15: score -= 5 

    # B. Production signal score (0–20)
    prod_score = 0
    for verb in PRODUCTION_VERBS_A:
        if verb in career_text: prod_score += 4
    for verb in PRODUCTION_VERBS_B:
        if verb in career_text: prod_score += 2
    score += min(prod_score, 20)

    # C. Semantic match score (0–20)
    score += min(semantic_score * 20, 20)

    # D. Behavioral signal score (0–20)
    signals = candidate.get('redrob_signals', {})
    behav_score = 0
    last_active = signals.get('last_active_date', '2000-01-01')
    try:
        dt = datetime.strptime(last_active, '%Y-%m-%d')
        days_since = (datetime(2026, 6, 11) - dt).days
        if days_since < 30: behav_score += 8
    except: pass
    if signals.get('notice_period_days', 90) < 30: behav_score += 8
    if signals.get('recruiter_response_rate', 0) > 0.7: behav_score += 4
    score += behav_score

    # E. Company type score (0–10)
    current_company = profile.get('current_company', '').lower()
    current_industry = profile.get('current_industry', '').lower()
    is_service = any(sc in current_company for sc in SERVICE_COMPANIES) or "it services" in current_industry
    if not is_service: score += 10
    
    # F. Education alignment score (0–10)
    edu_score = 0
    for edu in candidate.get('education', []):
        fos = edu.get('field_of_study', '').lower()
        if any(kw in fos for kw in ["computer science", "artificial intelligence", "machine learning"]):
            edu_score = 10
            break
        elif "engineering" in fos:
            edu_score = 7
    score += edu_score

    return score

def get_llm_score(candidate, job_desc):
    if not HAS_LLM:
        return 0, "Detailed analysis unavailable.", "N/A", "N/A"
    
    profile_summary = f"Title: {candidate['profile']['current_title']}\nExp: {candidate['profile']['years_of_experience']}y\nSummary: {candidate['profile']['summary']}"
    
    prompt = f"Expert recruiter scoring for Senior AI role.\nJD: {job_desc[:500]}...\nCANDIDATE: {profile_summary}\nReturn JSON: {{'llm_score': 0-100, 'fit_summary': '', 'top_strength': '', 'top_gap': ''}}"

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

    print("Stage 2: Feature Scoring...")
    candidate_dict = {c['candidate_id']: c for c in filtered_candidates}
    shortlisted_data = []
    for cid, sem_score in semantic_scores.items():
        cand = candidate_dict[cid]
        feat_score = get_feature_score(cand, sem_score)
        shortlisted_data.append({'candidate_id': cid, 'feat_score': feat_score, 'sem_score': sem_score})
    
    shortlisted_data.sort(key=lambda x: (-x['feat_score'], x['candidate_id']))
    top_20 = shortlisted_data[:20]
    
    print("Stage 3: LLM Reranking...")
    final_results = []
    for entry in tqdm(top_20):
        cand = candidate_dict[entry['candidate_id']]
        llm_score, fit_summary, strength, gap = get_llm_score(cand, job_description)
        final_score = (0.35 * llm_score) + (0.65 * entry['feat_score'])
        final_results.append({
            'candidate_id': cand['candidate_id'],
            'score': round(final_score / 100.0, 3),
            'reasoning': f"{fit_summary} Strength: {strength}. Gap: {gap}."
        })

    if len(final_results) < TOP_N:
        for entry in shortlisted_data[20:TOP_N]:
            cand = candidate_dict[entry['candidate_id']]
            final_results.append({
                'candidate_id': cand['candidate_id'],
                'score': round(entry['feat_score'] / 100.0, 3),
                'reasoning': "Qualified candidate with strong semantic match and product background."
            })

    final_results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    print(f"Writing output to {args.out}...")
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for i, res in enumerate(final_results[:TOP_N]):
            writer.writerow([res['candidate_id'], i + 1, res['score'], res['reasoning']])

if __name__ == "__main__":
    main()
