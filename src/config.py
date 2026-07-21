# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAG_DIR  = DATA_DIR / 'rag'
GOLD_DIR = DATA_DIR / 'gold'

# API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Model Names
EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
RERANKER_MODEL   = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
GROQ_MODEL       = 'llama-3.1-8b-instant'

# RAG Settings
K_RETRIEVAL = 15
K_FINAL     = 5
BM25_WEIGHT = 0.4
VEC_WEIGHT  = 0.6