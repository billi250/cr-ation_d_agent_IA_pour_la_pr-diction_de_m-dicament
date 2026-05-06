"""Petit test d'installation avant de lancer le RAG."""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Test du modèle d'embedding...")
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
vector = model.encode("Test d'embedding", normalize_embeddings=True)
print(f"Embedding OK — dimension : {len(vector)}")

print("Test FAISS...")
index = faiss.IndexFlatIP(len(vector))
index.add(np.array([vector], dtype=np.float32))
print(f"FAISS OK — {index.ntotal} vecteur(s) indexé(s)")

print("Installation OK.")
