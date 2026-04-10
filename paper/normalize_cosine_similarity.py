from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load data and drop any unnamed index column
df = pd.read_csv("evalset_embedding_similarities_cosine_top3.tsv", sep="\t")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Scale 'Similarity' to 0–100 and replace it in-place
scaler = MinMaxScaler(feature_range=(0, 100))
df["Similarity"] = scaler.fit_transform(df["Similarity"].values.reshape(-1, 1))

# Optionally round for readability
df["Similarity"] = df["Similarity"].round(2)

# Save output without index
df.to_csv("cosine_similarity_scaled_3.tsv", sep="\t", index=False)

#######################

df = pd.read_csv("evalset_embedding_similarities_cosine_top5.tsv", sep="\t")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Scale 'Similarity' to 0–100 and replace it in-place
scaler = MinMaxScaler(feature_range=(0, 100))
df["Similarity"] = scaler.fit_transform(df["Similarity"].values.reshape(-1, 1))

# Optionally round for readability
df["Similarity"] = df["Similarity"].round(2)

# Save output without index
df.to_csv("cosine_similarity_scaled_5.tsv", sep="\t", index=False)

########################

df = pd.read_csv("evalset_embedding_similarities_cosine_top1.tsv", sep="\t")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Scale 'Similarity' to 0–100 and replace it in-place
scaler = MinMaxScaler(feature_range=(0, 100))
df["Similarity"] = scaler.fit_transform(df["Similarity"].values.reshape(-1, 1))

# Optionally round for readability
df["Similarity"] = df["Similarity"].round(2)

# Save output without index
df.to_csv("cosine_similarity_scaled_1.tsv", sep="\t", index=False)

########################

df = pd.read_csv("evalset_embedding_similarities_cosine_top1_unweighted.tsv", sep="\t")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Replace 'Similarity' 1.0 100
df['Similarity'] = df['Similarity'].replace(1.0, 100)

# Save output without index
df.to_csv("cosine_similarity_scaled_1_unweighted.tsv", sep="\t", index=False)
