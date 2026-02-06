from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. Embedding model (~440MB download)
print("Downloading embedding model...")
embed = SentenceTransformer("BAAI/bge-base-en-v1.5")
embed.save(r"C:\Users\Lee\Desktop\offline_models\bge-base-en-v1.5")
print("Saved embedding model.")

# 2. Reranker model (~90MB download)
print("Downloading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker.save(r"C:\Users\Lee\Desktop\offline_models\ms-marco-MiniLM-L-6-v2")
print("Saved reranker model.")