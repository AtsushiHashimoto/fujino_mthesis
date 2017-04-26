# _*_ coding : utf-8 -*-

"""
FlowGraphのtreeファイルから共起頻度を計算
入力: viob2ファイル
        各行 : レシピ1stepの文章
               スペース区切りで　word/iob2tag
      出力保存先ディレクトリ
出力: co_occurrence.pickle
        keywordsのリスト
        各keywordsの頻度
        共起頻度を表す行列(インデックスは上記のリスト順)　
"""

import os
import argparse
import glob
import itertools
import pickle
import codecs
import pandas as pd
import numpy as np


RECIPE = 1715343


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help=u'入力ファイル')
    parser.add_argument('synonym_path', help=u'料理オントロジーファイル')
    parser.add_argument('output_dir', help=u'出力ディレクトリ')
    parser.add_argument('-t', '--tags', help='使用するタグ(tag1,tag2,...)', type=str, default='F,T,D,Q,Ac,Af,Sf,St')
    params = parser.parse_args()

    return vars(params)


def read_synonym(synonym_path):
    synonyms = pd.read_csv(synonym_path, delimiter='\t', header = None, encoding='utf-8')
    idx = list((synonyms.iloc[:, 0] != u'調理器具') & (synonyms.iloc[:, 0] != u'動作'))
    ontology = dict(zip(synonyms.iloc[idx, 2], synonyms.iloc[idx, 1]))
    return ontology


def parse_keywords(line, ontology, used_tags):
    keywords_by_step = []
    for token in line:
        token = token.split('/')
        if len(token) == 2:
            word = token[0]
            iob2_format = token[1].split('-')
            if len(iob2_format) == 2:
                recipe_tag = iob2_format[0]
                iob2_tag = iob2_format[1]
                if recipe_tag in used_tags:
                    if iob2_tag == 'B':
                        keywords_by_step.append(word)
                    elif iob2_tag == 'I':
                        keywords_by_step[-1] += word 
    # ontologyに入っているもののみ
    keywords_by_step = [ontology[keyword] for keyword in keywords_by_step if keyword in ontology]
    keywords_by_step = list(set(keywords_by_step)) # 重複を除く
    return keywords_by_step


def extract_keywords(line, recipe_no, keywords, tmp_keywords, occur, ontology, used_tags):
    keywords_by_step = []
    line = line.split()
    if len(line) == 0: #レシピ終了
        for kwd in tmp_keywords:
            occur[keywords.index(kwd)] += 1
        tmp_keywords = []
        recipe_no += 1
    else:
        keywords_by_step = parse_keywords(line, ontology, used_tags)
        for kwd in keywords_by_step: # 重複を除く
            if kwd not in keywords: # 初めて登場したレシピ用語
                keywords.append(kwd)
                tmp_keywords.append(kwd)
                occur.append(0)
            elif kwd not in tmp_keywords: # レシピ中で初めて登場したレシピ用語
                tmp_keywords.append(kwd)
    return recipe_no, keywords, tmp_keywords, occur


def count_cooccurrence(line, recipe_no, keywords, tmp_keywords, cooccur, ontology, used_tags):
    keywords_by_step = []
    line = line.split()
    if len(line) == 0:
        for kwd1, kwd2 in itertools.combinations(tmp_keywords, 2):
            if kwd1 in keywords and kwd2 in keywords: 
                idx1 = keywords.index(kwd1)
                idx2 = keywords.index(kwd2)
                cooccur[idx1, idx2] += 1
                cooccur[idx2, idx1] += 1
        tmp_keywords = []
        recipe_no += 1
    else:
        keywords_by_step = parse_keywords(line, ontology, used_tags)
        for kwd in keywords_by_step:
            if kwd not in tmp_keywords:
                tmp_keywords.append(kwd) # レシピ中で初めて登場したレシピ用語
    return recipe_no, cooccur, tmp_keywords


def main(params):
    input_path = params['input_path']
    synonym_path = params['synonym_path']
    output_dir = params['output_dir']
    used_tags = params['tags']

    used_tags = used_tags.split(',')
    print(used_tags)

    tag_str = ''
    for t in used_tags:
        tag_str += t

    ontology = read_synonym(synonym_path)

    # レシピに出現するキーワードを全て取得
    output_file = os.path.join(output_dir, 'viob2_keywords_%s.pickle' % tag_str)
    if os.path.exists(output_file):
        with open(output_file, 'rb') as fin:
            keywords, occur, recipe_no = pickle.load(fin)
    else:
        keywords = []
        occur = []
        tmp_keywords = []
        recipe_no = 0
        with codecs.open(input_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if recipe_no % 1000 == 0:
                    print("extract keywords... %d     \r"%recipe_no, end='') 
                recipe_no, keywords, tmp_keywords, occur =\
                    extract_keywords(line, recipe_no, keywords, tmp_keywords, occur, ontology, used_tags)
        with open(output_file, 'wb') as fout:
            pickle.dump((keywords, occur, recipe_no), fout, protocol=0)


    # 行列を拡張していくのはコストが高いのでキーワード数を数えてから共起回数を数える
    i = 0
    tmp_keywords = []
    cooccur = np.zeros((len(keywords), len(keywords)))
    with codecs.open(input_path, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            if i % 1000 == 0:
                print("count co-occurrence... %d/%d     \r"%(i, recipe_no), end='') 
            i, cooccur, tmp_keywords =\
                count_cooccurrence(line, i, keywords, tmp_keywords, cooccur, ontology, used_tags)

    print ("keywords:%d     "%len(keywords))
    print (keywords[0:10])
    print (occur[0:10])
    print ("cooccur ", cooccur.shape)
    print (cooccur[0:10, 0:10])

    with open(os.path.join(output_dir, 'viob2_cooccurence_%s.pickle' % tag_str), 'wb') as fout:
        pickle.dump((keywords, occur, cooccur, recipe_no), fout, protocol=0)


if __name__ == '__main__':
    params = parse()
    main(params)

