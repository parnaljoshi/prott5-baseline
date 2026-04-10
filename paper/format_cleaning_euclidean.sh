sed -ri 's/sp\|//g' euclidean_similarity_normalized_3.tsv
sed -ri 's/tr\|//g' euclidean_similarity_normalized_3.tsv
sed -ri 's/\|[^\t]*\t/\t/g' euclidean_similarity_normalized_3.tsv

sed -ri 's/sp\|//g' euclidean_similarity_normalized_5.tsv
sed -ri 's/tr\|//g' euclidean_similarity_normalized_5.tsv
sed -ri 's/\|[^\t]*\t/\t/g' euclidean_similarity_normalized_5.tsv

sed -ri 's/sp\|//g' euclidean_similarity_normalized_1.tsv
sed -ri 's/tr\|//g' euclidean_similarity_normalized_1.tsv
sed -ri 's/\|[^\t]*\t/\t/g' euclidean_similarity_normalized_1.tsv

sed -ri 's/sp\|//g' euclidean_similarity_normalized_1_unweighted.tsv
sed -ri 's/tr\|//g' euclidean_similarity_normalized_1_unweighted.tsv
sed -ri 's/\|[^\t]*\t/\t/g' euclidean_similarity_normalized_1_unweighted.tsv
