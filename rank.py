import json
import csv
import heapq
import os
from datetime import datetime

# Configuration
CANDIDATES_FILE = "data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
OUTPUT_FILE = "team_gemini.csv"
TOP_N = 100

# Keywords for scoring
EMBEDDINGS_KWS = ["embeddings", "sentence-transformers", "bge", "e5", "bert", "cross-encoder", "bi-encoder", "semantic search"]
VEC_DB_KWS = ["pinecone", "milvus", "weaviate", "qdrant", "opensearch", "elasticsearch", "faiss", "vector database", "vector search", "chroma", "lancedb"]
RANKING_KWS = ["ndcg", "mrr", "map", "ranking", "retrieval", "rerank", "bm25", "hybrid search", "ltr", "learning to rank", "xgboost", "lightgbm", "recommender"]
LLM_KWS = ["llm", "gpt", "claude", "llama", "fine-tuning", "lora", "qlora", "peft", "rag", "langchain", "llama-index"]
PYTHON_KWS = ["python", "pytorch", "tensorflow", "numpy", "pandas", "scikit-learn"]

SERVICE_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini", 
    "hcl", "tech mahindra", "mindtree", "l&t", "larsen & toubro", "persistent", "zensar", 
    "hexaware", "mphasis", "ust global", "ntt data", "fujitsu", "ibm", "deloitte", "ey", 
    "kpmg", "pwc", "capita", "atos", "conduent", "genpact"
}

PRODUCT_KEYWORDS = ["product", "saas", "tech", "platform", "startup", "labs"]

def is_service_company(company, industry):
    if not company: return False
    c_lower = company.lower()
    if any(sc in c_lower for sc in SERVICE_COMPANIES):
        return True
    if industry and "it services" in industry.lower():
        # Double check if it's a known product company that might be misclassified
        return True
    return False

def check_honeypot(candidate):
    # 1. Expert with 0 months
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            return True
            
    # 2. Years of experience vs career history
    total_months = sum(job.get('duration_months', 0) for job in candidate.get('career_history', []))
    claimed_years = candidate.get('profile', {}).get('years_of_experience', 0)
    if total_months / 12 > claimed_years + 3: # Buffer for overlap
        return True
    
    # 3. Impossible experience
    if claimed_years > 45: return True
    
    # 4. Education sequence check (e.g. Masters before Bachelors)
    edu_list = candidate.get('education', [])
    if len(edu_list) > 1:
        # Check for M.Tech/PhD before B.Tech
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

    return False

def get_score(candidate):
    score = 0
    reasons = []
    
    profile = candidate['profile']
    summary = profile.get('summary', '').lower()
    headline = profile.get('headline', '').lower()
    skills_list = [s['name'].lower() for s in candidate.get('skills', [])]
    skill_names_str = " ".join(skills_list)
    
    all_text = f"{headline} {summary} {skill_names_str}"
    
    # 1. Experience Level (5-9 years is ideal)
    yoe = profile.get('years_of_experience', 0)
    if 5 <= yoe <= 9:
        score += 20
        reasons.append(f"{yoe}y experience")
    elif 4 <= yoe <= 12:
        score += 10
    else:
        score += 2
        
    # 2. Keyword matching with weights
    weights = {
        "Embeddings": (EMBEDDINGS_KWS, 12),
        "Vector DB": (VEC_DB_KWS, 12),
        "Ranking": (RANKING_KWS, 15),
        "LLM": (LLM_KWS, 8),
        "Python": (PYTHON_KWS, 5)
    }
    
    matched_cats = []
    for cat, (kws, weight) in weights.items():
        if any(kw in all_text for kw in kws):
            score += weight
            matched_cats.append(cat)
            
    if matched_cats:
        reasons.append("/".join(matched_cats))
            
    # Production experience mention
    if "production" in all_text or "deployed" in all_text or "scale" in all_text:
        score += 8
        reasons.append("Production experience")
        
    # 3. Product vs Service
    history = candidate.get('career_history', [])
    product_count = 0
    service_only = True
    for job in history:
        if not is_service_company(job.get('company'), job.get('industry')):
            product_count += 1
            service_only = False
    
    if not service_only:
        score += 15
        if product_count > 1:
            score += 5
        reasons.append("Product company background")
    else:
        score -= 20 # Strong penalty for pure service background
        
    # 4. Title Relevance
    current_title = profile.get('current_title', '').lower()
    RELEVANT_TITLES = ["engineer", "developer", "architect", "lead", "cto", "tech", "data", "ml", "ai"]
    if not any(t in current_title for t in RELEVANT_TITLES):
        score -= 30 # Trap: Marketing, Sales, etc.
    
    if "research" in current_title or "scientist" in current_title:
        score -= 5 # researcher penalty
    elif "engineer" in current_title:
        score += 5 # engineer bonus
            
    # 5. Location & Relocation
    location = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    
    # Pune/Noida are preferred
    pref_cities = ["pune", "noida", "greater noida"]
    # Tier-1 cities
    tier1_cities = ["bangalore", "bengaluru", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "chennai", "kolkata", "ahmedabad"]
    
    if country == "india" or any(city in location for city in pref_cities + tier1_cities):
        score += 5
        if any(city in location for city in pref_cities):
            score += 10
            reasons.append("Local in Pune/Noida")
        elif any(city in location for city in tier1_cities):
            score += 5
            reasons.append("Tier-1 city resident")
        
        if candidate.get('redrob_signals', {}).get('willing_to_relocate'):
            score += 5
            reasons.append("Willing to relocate")
            
    # 6. Behavioral Signals
    signals = candidate.get('redrob_signals', {})
    
    # Recency
    last_active = signals.get('last_active_date', '2000-01-01')
    try:
        dt = datetime.strptime(last_active, '%Y-%m-%d')
        days_since = (datetime(2026, 6, 11) - dt).days
        if days_since < 30:
            score += 10
            reasons.append("Active recently")
        elif days_since < 90:
            score += 4
    except: pass
    
    # Engagement
    if signals.get('recruiter_response_rate', 0) > 0.7:
        score += 8
        reasons.append("High engagement")
        
    # Notice Period
    notice = signals.get('notice_period_days', 90)
    if notice <= 30:
        score += 10
        reasons.append("Immediate joiner")
    elif notice <= 60:
        score += 5
    elif notice > 90:
        score -= 10 # Slow notice period penalty
        
    # Github
    if signals.get('github_activity_score', 0) > 50:
        score += 7
        reasons.append("Strong GitHub")
        
    # 7. Evaluation Frameworks
    eval_terms = ["ndcg", "mrr", "map", "evaluation framework", "a/b test", "offline benchmark"]
    if any(term in all_text for term in eval_terms):
        score += 10
        reasons.append("Eval framework expert")

    # Construct reasoning
    if not reasons:
        reasoning = "Matches core technical requirements for the Senior AI Engineer role."
    else:
        # Construct a more natural sentence
        # Pick top 4 reasons
        unique_reasons = []
        for r in reasons:
            if r not in unique_reasons:
                unique_reasons.append(r)
        
        reasoning = "Excellent candidate with " + ", ".join(unique_reasons[:3])
        if len(unique_reasons) > 3:
            reasoning += f", and {unique_reasons[3]}"
        reasoning += "."
        
    return score, reasoning

import argparse

def main():
    parser = argparse.ArgumentParser(description='Rank candidates for Senior AI Engineer role.')
    parser.add_argument('--candidates', type=str, default="data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl",
                        help='Path to candidates.jsonl file')
    parser.add_argument('--out', type=str, default="1.csv",
                        help='Path to output CSV file')
    args = parser.parse_args()

    candidates_file = args.candidates
    output_file = args.out
    
    top_candidates = []
    
    count = 0
    if not os.path.exists(candidates_file):
        print(f"Error: {candidates_file} not found.")
        return

    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 20000 == 0:
                print(f"Processed {count} candidates...")
            
            candidate = json.loads(line)
            
            if check_honeypot(candidate):
                continue
            
            score, reasoning = get_score(candidate)
            
            if score > 20: 
                top_candidates.append({
                    'candidate_id': candidate['candidate_id'],
                    'score': score,
                    'reasoning': reasoning
                })
                
    print(f"Sorting {len(top_candidates)} candidates...")
    top_candidates.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    final_top = top_candidates[:TOP_N]
    
    print(f"Writing {len(final_top)} candidates to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for i, cand in enumerate(final_top):
            score_out = cand['score'] / 100.0
            writer.writerow([cand['candidate_id'], i + 1, score_out, cand['reasoning']])

if __name__ == "__main__":
    main()
