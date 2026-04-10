import h5py
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

print("Start:", time.ctime())

# Load DB embeddings
with h5py.File('../embeddings/blast_db_embeddings.h5', 'r') as db_file:
    db_ids = list(db_file.keys())
    Y = np.vstack([db_file[db][:] for db in tqdm(db_ids, desc="Loading DB embeddings")])

# Load query embeddings
with h5py.File('../embeddings/evalset_embeddings.h5', 'r') as query_file:
    query_ids = list(query_file.keys())
    X = np.vstack([query_file[qid][:] for qid in tqdm(query_ids, desc="Loading Query embeddings")])

# Compute Cosine similarity matrix
print("Computing Cosine similarity matrix...", flush=True)
sim_matrix = cosine_similarity(X, Y)  # Shape: (num_queries, num_db)

# Extract top-K most similar entries (highest cosine similarity)
def extract_top_k(sim_matrix, k, query_ids, db_ids):
    top_indices = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]  # Get top k most similar (highest similarity)
    top_similarities = np.take_along_axis(sim_matrix, top_indices, axis=1)

    # Sort by similarity in descending order
    sorted_order = np.argsort(top_similarities, axis=1)[:, ::-1]  # Sort each row in descending order
    top_indices_sorted = np.take_along_axis(top_indices, sorted_order, axis=1)
    top_similarities_sorted = np.take_along_axis(top_similarities, sorted_order, axis=1)

    results = []
    for i, qid in enumerate(query_ids):
        for j in range(k):
            db_idx = top_indices_sorted[i, j]
            results.append((qid, db_ids[db_idx], 0, 0, top_similarities_sorted[i, j], 0))
    return pd.DataFrame(results, columns=["Query ID", "DB ID", "e-val", "Length", "Cosine Similarity", "N-ident"])

# Generate top-3 and top-5 most similar entries
print("Extracting closest 3 entries...", flush=True)
super_df_3 = extract_top_k(sim_matrix, 3, query_ids, db_ids)
super_df_3.to_csv("evalset_embedding_similarities_cosine_top3.tsv", sep="\t", index=False)

print("Extracting closest 5 entries...", flush=True)
super_df_5 = extract_top_k(sim_matrix, 5, query_ids, db_ids)
super_df_5.to_csv("evalset_embedding_similarities_cosine_top5.tsv", sep="\t", index=False)

print("Extracting closest 1 entry...", flush=True)
super_df_1 = extract_top_k(sim_matrix, 1, query_ids, db_ids)
super_df_1.to_csv("evalset_embedding_similarities_cosine_top1.tsv", sep="\t", index=False)

print("Done:", time.ctime())
