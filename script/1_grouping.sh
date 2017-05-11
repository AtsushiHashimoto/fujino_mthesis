mkdir -p ../exp/grouping
python3 ../tools/grouping/python3/count_co-occurrence_viob2.py ../external/cookpad_step_grep_recipesp_norm_sentsp.Viob2 ../external/ontology/synonym.tsv ../exp/grouping/ -t F
python3 ../tools/grouping/python3/clustering_of_exclusive_keywords.py ../exp/grouping/viob2_cooccurence_F.pickle ../exp/grouping/



