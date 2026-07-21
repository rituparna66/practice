import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import faiss
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────
# STEP 1 — LOAD EMBEDDING MODEL
# ─────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded\n")


# ─────────────────────────────────────────────
# STEP 2 — SINGLE SENTENCE EMBEDDING DEMO
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 2 — Single Sentence Embedding")
print("=" * 55)

sentence  = "Gold price surged due to inflation fears"
embedding = model.encode(sentence)

print(f"Sentence : {sentence}")
print(f"Shape    : {embedding.shape}")        # (384,)
print(f"Sample   : {embedding[:5].round(4)}")
print()
