import h5py
from sklearn.metrics.pairwise import euclidean_distances
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

# Compute Euclidean distance matrix
print("Computing Euclidean distance matrix...", flush=True)
dist_matrix = euclidean_distances(X, Y)  # Shape: (num_queries, num_db)

# Extract top-K smallest distances
def extract_top_k(dist_matrix, k, query_ids, db_ids):
    top_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    top_distances = np.take_along_axis(dist_matrix, top_indices, axis=1)

    sorted_order = np.argsort(top_distances, axis=1)
    top_indices_sorted = np.take_along_axis(top_indices, sorted_order, axis=1)
    top_distances_sorted = np.take_along_axis(top_distances, sorted_order, axis=1)

    results = []
    for i, qid in enumerate(query_ids):
        for j in range(k):
            db_idx = top_indices_sorted[i, j]
            results.append((qid, db_ids[db_idx], 0, 0, top_distances_sorted[i, j], 0))
    return pd.DataFrame(results, columns=["Query ID", "DB ID", "e-val", "Length", "Distance", "N-ident"])

# Generate top-3 and top-5
print("Extracting closest 3 entries...", flush=True)
super_df_3 = extract_top_k(dist_matrix, 3, query_ids, db_ids)
super_df_3.to_csv("evalset_embedding_distances_euclidean_top3.tsv", sep="\t", index=False)

print("Extracting closest 5 entries...", flush=True)
super_df_5 = extract_top_k(dist_matrix, 5, query_ids, db_ids)
super_df_5.to_csv("evalset_embedding_distances_euclidean_top5.tsv", sep="\t", index=False)

print("Extracting closest 1 entry...", flush=True)
super_df_1 = extract_top_k(dist_matrix, 1, query_ids, db_ids)
super_df_1.to_csv("evalset_embedding_distances_euclidean_top1.tsv", sep="\t", index=False)

print("Done:", time.ctime())
