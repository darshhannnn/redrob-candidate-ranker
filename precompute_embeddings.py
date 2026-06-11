import json
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# This script is part of the pre-computation phase and is allowed to exceed 5 minutes.
# It populates the ChromaDB index for all 100k candidates.

SEMANTIC_MODEL_NAME = 'all-mpnet-base-v2'
CHROMA_PATH = "./chroma_db"

def check_honeypot(candidate):
    # (Same integrity checks as in rank.py to ensure the index is clean)
    # 1. Expert with 0 months
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            return True
            
    # 2. Years of experience vs career history
    total_months = sum(job.get('duration_months', 0) for job in candidate.get('career_history', []))
    claimed_years = candidate.get('profile', {}).get('years_of_experience', 0)
    if total_months / 12 > claimed_years + 3: return True
    
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
        if b_year and m_year and m_year < b_year: return True

    return False

def main():
    parser = argparse.ArgumentParser(description='Pre-populate ChromaDB for candidates')
    parser.add_argument('--candidates', type=str, required=True, help='Path to candidates.jsonl')
    args = parser.parse_args()

    print("Loading model...")
    model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="candidates")
    
    # Load all candidates
    print("Reading candidates...")
    candidates = []
    with open(args.candidates, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            if not check_honeypot(cand):
                candidates.append(cand)
    
    print(f"Total candidates to embed: {len(candidates)}")
    
    # Batch embedding
    batch_size = 128
    for i in tqdm(range(0, len(candidates), batch_size)):
        batch = candidates[i:i+batch_size]
        texts = []
        ids = []
        metadatas = []
        for c in batch:
            text = f"{c['profile']['current_title']} {c['profile']['summary']} {' '.join([s['name'] for s in c['skills']])}"
            texts.append(text)
            ids.append(c['candidate_id'])
            # Store some basic metadata to avoid full json lookup if possible (optional)
            metadatas.append({"cid": c['candidate_id']})
        
        embeddings = model.encode(texts).tolist()
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print(f"ChromaDB populated at {CHROMA_PATH}")

if __name__ == "__main__":
    main()
