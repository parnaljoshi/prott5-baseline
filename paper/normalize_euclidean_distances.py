import pandas as pd

def norm_sim(x):
    x = (x_max - x)/x_max
    sim_per = x*100
    return round(sim_per, 2)

df = pd.read_csv("evalset_embedding_distances_euclidean_top3.tsv", sep="\t")
x_max = df["Distance"].max()

# Replace the "Distance" column with the normalized "Similarity" values
df["Distance"] = df["Distance"].map(norm_sim)

# Rename the "Distance" column to "Similarity"
df.rename(columns={"Distance": "Similarity"}, inplace=True)

# Save the modified DataFrame to a new file with the "Similarity" column in the same place
df.to_csv("euclidean_similarity_normalized_3.tsv", sep="\t", index=False)

###########################

df = pd.read_csv("evalset_embedding_distances_euclidean_top5.tsv", sep="\t")
x_max = df["Distance"].max()

# Replace the "Distance" column with the normalized "Similarity" values
df["Distance"] = df["Distance"].map(norm_sim)

# Rename the "Distance" column to "Similarity"
df.rename(columns={"Distance": "Similarity"}, inplace=True)

# Save the modified DataFrame to a new file with the "Similarity" column in the same place
df.to_csv("euclidean_similarity_normalized_5.tsv", sep="\t", index=False)

##########################

df = pd.read_csv("evalset_embedding_distances_euclidean_top1.tsv", sep="\t")
x_max = df["Distance"].max()

# Replace the "Distance" column with the normalized "Similarity" values
df["Distance"] = df["Distance"].map(norm_sim)

# Rename the "Distance" column to "Similarity"
df.rename(columns={"Distance": "Similarity"}, inplace=True)

# Save the modified DataFrame to a new file with the "Similarity" column in the same place
df.to_csv("euclidean_similarity_normalized_1.tsv", sep="\t", index=False)

##########################

df = pd.read_csv("evalset_embedding_distances_euclidean_top1_unweighted.tsv", sep="\t")

# Replace the "Distance" column with the normalized "Similarity" values
df["Distance"] = df["Distance"].map(norm_sim)
df['Distance'] = df['Distance'].replace(0.0, 100)

# Rename the "Distance" column to "Similarity"
df.rename(columns={"Distance": "Similarity"}, inplace=True)

# Save the modified DataFrame to a new file with the "Similarity" column in the same place
df.to_csv("euclidean_similarity_normalized_1_unweighted.tsv", sep="\t", index=False)
