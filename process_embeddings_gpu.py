import h5py
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import time
from scipy.spatial import distance
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# your_outputs = Parallel(n_jobs=32, verbose=10)(delayed(yourfxn)(your_inputs_i) for your_inputs_i in list_of_things)
# Parallel(n_jobs=num_cores)(delayed(myfunction)(i,parameters) for i in inputs)

print(time.ctime())
query_embeddings = h5py.File('embeddings/evalset_embeddings.h5')
#query_embeddings = h5py.File('test_embeddings.h5')
X = []
test_ids = []
Y = []
db_ids = []

db_embeddings = h5py.File('embeddings/blast_db_embeddings.h5')
#db_embeddings = h5py.File('test_embeddings.h5')
for db in tqdm(db_embeddings.keys()):
    # print(db, db_embeddings[db][:])
    db_ids.append(db)
    Y.append(db_embeddings[db][:])
    time.sleep(0.1)

#fht = open("newset_df_dist_gpu.tsv", "w")
df = pd.DataFrame()
#temp_df = pd.DataFrame()
super_df = [] 
for i in tqdm(range(0, len(query_embeddings.keys()))):
    X = query_embeddings[list(query_embeddings.keys())[i]][:]
    test_ids = list(query_embeddings.keys())[i]
    d = pd.DataFrame(None, columns=["Query ID", "DB ID", "e-val", "Length", "Similarity", "N-ident"])
    #sim_matrix = 1 - euclidean_distances(X.reshape(1, -1), Y)
    #sim_matrix = sim_matrix * 100
    sim_matrix = euclidean_distances(X.reshape(1, -1), Y)
    d = pd.concat([d, pd.DataFrame(list(zip(np.repeat(test_ids, len(db_ids)), db_ids, np.repeat(0, len(db_ids)),
                                            np.repeat(0, len(db_ids)), np.transpose(sim_matrix[0]),
                                            np.repeat(0, len(db_ids)))),
                                   columns=["Query ID", "DB ID", "e-val", "Length", "Similarity", "N-ident"])])
    #df = pd.concat([df, d.nlargest(250, 'Similarity')])
    #fht.write(str(d.nlargest(250, 'Similarity'))+"\n")
    #df = pd.concat([df, d.nsmallest(250, 'Similarity')])
    #temp_df = pd.concat([temp_df, d])
    #temp_df.to_csv("newset_df_dist_gpu.tsv", sep="\t")
    #fht.write(str(d)+"\n")
    #super_df.append(d.nsmallest(250, 'Similarity'))
    super_df.append(d.nsmallest(1000, 'Similarity'))
    time.sleep(0.1)
#print(df)

super_df = pd.concat(super_df, axis=0)
super_df.to_csv("evalset_embedding_distances_denorm_1000.tsv", sep="\t")

#df.to_csv("newset_embedding_distances_denorm_gpu.tsv", sep="\t")
#temp_df.to_csv("newset_df_dist_gpu.tsv", sep="\t")

#x_max = df["Similarity"].max()
#def norm_sim(x):
#    x = (x_max - x)/x_max
#    sim_per = x*100
#    return round(sim_per, 2)

#df["Similarity"] = df["Similarity"].map(norm_sim)
#df.to_csv("eval_embedding_similarity_singular.tsv", sep="\t")

#fht.close()

print(time.ctime())
