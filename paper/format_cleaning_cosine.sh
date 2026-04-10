sed -ri 's/sp\|//g' cosine_similarity_scaled_3.tsv
sed -ri 's/tr\|//g' cosine_similarity_scaled_3.tsv
sed -ri 's/\|[^\t]*\t/\t/g' cosine_similarity_scaled_3.tsv

sed -ri 's/sp\|//g' cosine_similarity_scaled_5.tsv
sed -ri 's/tr\|//g' cosine_similarity_scaled_5.tsv
sed -ri 's/\|[^\t]*\t/\t/g' cosine_similarity_scaled_5.tsv

sed -ri 's/sp\|//g' cosine_similarity_scaled_1.tsv
sed -ri 's/tr\|//g' cosine_similarity_scaled_1.tsv
sed -ri 's/\|[^\t]*\t/\t/g' cosine_similarity_scaled_1.tsv

sed -ri 's/sp\|//g' cosine_similarity_scaled_1_unweighted.tsv
sed -ri 's/tr\|//g' cosine_similarity_scaled_1_unweighted.tsv
sed -ri 's/\|[^\t]*\t/\t/g' cosine_similarity_scaled_1_unweighted.tsv
