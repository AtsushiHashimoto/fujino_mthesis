# _*_ coding: utf-8 -*-

import pandas as pd

def load_food_synonym(synonym_path, key="swing", seasoning=True):
    """
    input
    synonym_path: full path of synonym.tsv
    key: setting of dictionary of result; "swing" swing->concept  "concept" concept->swing
    seasoning: use seasoning?

    output
    dictionay of synonym
    """
    synonyms = pd.read_csv(synonym_path, delimiter='\t', header = None, encoding='utf-8')
    not_used = [u"動作", u"調理器具"]
    if not seasoning:
        not_used.append(u"調味料")
    idx = [l not in not_used for l in list(synonyms.iloc[:, 0])]
    assert True in idx and False in idx
    if key == "concept":
        synonym_dic = {} 
        for concept, swing in zip(synonyms.iloc[idx, 1], synonyms.iloc[idx, 2]):
            if concept not in synonym_dic:
                synonym_dic[concept] = []
            synonym_dic[concept].append(swing)
    elif key == "swing":
        synonym_dic = dict(zip(synonyms.iloc[idx, 2], synonyms.iloc[idx, 1]))
    else:
        raise ValueError("Argument 'key' is not 'swing' or 'concept'.: %s" % key)

    return synonym_dic


