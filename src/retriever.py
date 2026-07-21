# src/retriever.py
import json
import pickle
import re
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from src.config import (
    RAG_DIR, EMBED_MODEL_NAME, RERANKER_MODEL,
    K_RETRIEVAL, K_FINAL, BM25_WEIGHT, VEC_WEIGHT
)


class HybridRetriever:
    """BM25 + FAISS Vector + Cross-Encoder reranking"""

    def __init__(self):
        print('🔄 Loading retrieval components...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load chunks
        with open(RAG_DIR / 'chunks_store.json',
                  encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f'   ✅ Chunks: {len(self.chunks)}')

        # Load FAISS index
        self.index = faiss.read_index(
            str(RAG_DIR / 'heritage_index.faiss')
        )
        print(f'   ✅ FAISS: {self.index.ntotal} vectors')

        # Load BM25
        with open(RAG_DIR / 'bm25_index.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        print('   ✅ BM25 loaded')

        # Load embedding model
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        self.embed_model = self.embed_model.to(self.device)
        print(f'   ✅ Embed model on {self.device.upper()}')

        # Load reranker
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print('   ✅ Reranker loaded')

    def tokenize(self, text):
        return re.findall(r'[\w]+', text.lower())

    def retrieve(self, query, k_final=K_FINAL):

        # Vector search
        q_embed = self.embed_model.encode(
            [query], normalize_embeddings=True
        )
        vec_scores, vec_indices = self.index.search(
            q_embed.astype(np.float32), K_RETRIEVAL
        )

        # BM25 search
        tokens = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:K_RETRIEVAL]

        # RRF Fusion
        scores = {}
        K = 60
        for rank, idx in enumerate(vec_indices[0]):
            if idx >= 0:
                if idx not in scores:
                    scores[idx] = 0
                scores[idx] += VEC_WEIGHT * (1 / (K + rank + 1))

        for rank, idx in enumerate(bm25_top):
            if bm25_scores[idx] > 0:
                if idx not in scores:
                    scores[idx] = 0
                scores[idx] += BM25_WEIGHT * (1 / (K + rank + 1))

        # Priority boost
        for idx in scores:
            p = self.chunks[idx].get('priority', 3)
            if p == 1:
                scores[idx] += 0.02
            elif p == 2:
                scores[idx] += 0.01

        # Top candidates
        top_candidates = sorted(
            scores.keys(),
            key=lambda x: scores[x],
            reverse=True
        )[:K_RETRIEVAL]

        if not top_candidates:
            return []

        # Rerank
        texts = [
            self.chunks[i]['text'][:512]
            for i in top_candidates
        ]
        pairs = [[query, t] for t in texts]
        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(top_candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k_final]

        results = []
        for idx, score in reranked:
            chunk = self.chunks[idx].copy()
            chunk['retrieval_score'] = float(score)
            results.append(chunk)

        return results