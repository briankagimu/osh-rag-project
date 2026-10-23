import os
import re
import json
import time
import datetime
import hashlib
from pathlib import Path

# Use HuggingFace mirror for China
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
QPREFIX = 'Represent this sentence for searching relevant passages: '

# Tokenizer for BM25
tok = lambda s: re.findall(r'[a-z0-9]+', s.lower())

models = {}

def get_models():
    if not models:
        print("Loading embedding model...")
        models['embed'] = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print("Loading re-ranker model...")
        models['rerank'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return models['embed'], models['rerank']

def load_chunks(name='chunks.jsonl'):
    with open(ROOT / 'data' / 'processed' / name, encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion to combine dense and sparse results."""
    s = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst, start=1):
            s[cid] = s.get(cid, 0) + 1 / (k + rank)
    return sorted(s, key=s.get, reverse=True)

class Retriever:
    def __init__(self, chunks, name='osh_act'):
        self.chunks = chunks
        self.by_id = {c['id']: c for c in chunks}
        self.embed, self.reranker = get_models()
        
        texts = [c['embed_text'] for c in chunks]
        
        # Create a unique collection name based on the text hash
        cname = f"{name}_{hashlib.md5(''.join(texts).encode('utf-8')).hexdigest()[:8]}"
        client = chromadb.PersistentClient(path=str(ROOT / 'outputs' / 'chroma'))
        
        try:
            self.col = client.get_collection(cname)
            print("Loaded existing ChromaDB index.")
        except Exception:
            print("Creating new ChromaDB index...")
            emb = self.embed.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
            # FIX: metadata goes inside create_collection, not add
            self.col = client.create_collection(cname, metadata={'hnsw:space': 'cosine'})
            self.col.add(
                ids=[c['id'] for c in chunks],
                documents=[c['text'] for c in chunks],
                embeddings=emb.tolist()
            )
            
        self.bm25 = BM25Okapi([tok(t) for t in texts])

    def dense(self, q, k=10):
        qe = self.embed.encode([QPREFIX + q], normalize_embeddings=True).tolist()
        return self.col.query(query_embeddings=qe, n_results=k)['ids'][0]

    def sparse(self, q, k=10):
        top = self.bm25.get_scores(tok(q)).argsort()[::-1][:k]
        return [self.chunks[i]['id'] for i in top]

    def hybrid(self, q, rerank=True, pool=20, final=5):
        fused = rrf([self.dense(q, 10), self.sparse(q, 10)])[:pool]
        if not rerank:
            return fused[:final], None
            
        # Cross-encoder re-ranking
        sc = self.reranker.predict([(q, self.by_id[i]['embed_text']) for i in fused])
        order = sc.argsort()[::-1][:final]
        return [fused[i] for i in order], [float(sc[i]) for i in order]

    def search(self, q, mode='hybrid_rerank', k=10):
        if mode == 'dense': return self.dense(q, k)
        if mode == 'bm25': return self.sparse(q, k)
        if mode == 'hybrid': return self.hybrid(q, rerank=False, final=k)[0]
        return self.hybrid(q, rerank=True, final=k)[0]


    # --- PHASE 4: GENERATION ---
MODEL = 'gpt-4o-mini'
SYSTEM = (ROOT / 'prompts' / 'system_prompt_v1.txt').read_text(encoding='utf-8')
REFUSAL = 'I cannot answer this with sufficient confidence. Please consult the OSH Act directly.'
ABSTAIN_BELOW = -2.0  # Chosen from Phase 3 testing

# OhMyGPT prices are usually similar to OpenAI's. Update if needed.
PRICE_IN, PRICE_OUT = 0.15, 0.60 # USD per 1,000,000 tokens

_client, _retriever = None, None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever(load_chunks())
    return _retriever

def answer(q, retriever=None):
    global _client
    R = retriever or get_retriever()
    _client = _client or OpenAI() # Reads API key and Base URL from .env
    
    t0 = time.perf_counter()
    ids, scores = R.hybrid(q, rerank=True, final=5)
    
    out = {'q': q, 'ids': ids, 'scores': scores, 'tokens_in': 0, 'tokens_out': 0, 'cost_usd': 0.0}
    
    # Abstain if the top score is too low
    if not scores or scores[0] < ABSTAIN_BELOW:
        out.update(answer=REFUSAL, abstained=True)
    else:
        ctx = '\n\n'.join(f"[{n}] ({R.by_id[i]['citation']})\n{R.by_id[i]['text']}" for n, i in enumerate(ids, 1))
        
        r = _client.chat.completions.create(
            model=MODEL, 
            temperature=0, 
            messages=[
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f"Context:\n{ctx}\n\nQuestion: {q}"}
            ]
        )
        u = r.usage
        out.update(
            answer=r.choices[0].message.content, 
            abstained=False, 
            tokens_in=u.prompt_tokens, 
            tokens_out=u.completion_tokens, 
            cost_usd=(u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT) / 1e6
        )
        
    out['latency_s'] = round(time.perf_counter() - t0, 2)
    out['time'] = datetime.datetime.now().isoformat()
    
    # Log the query
    logdir = ROOT / 'outputs' / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    with open(logdir / 'queries.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(out, ensure_ascii=False) + '\n')
        
    return out
