# _*_ coding: utf-8 -*-
# Python 3.x

"""
入力: cooccurrence.pickle 
      出力保存先ディレクトリ
出力: clustering_keywords.pickle
        keywordsのリスト
        各keywordsの頻度
        共起頻度を表す行列(インデックスは上記のリスト順)　
        レシピ数
"""

import os
import csv
import pickle
import argparse
import itertools

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('cooccurrence_path', help=u'cooccurrence.pickle')
    parser.add_argument('output_dir', help=u'出力ディレクトリ')

    params = parser.parse_args()

    return vars(params)

def estimate_cooccur_prob(keywords, occur, cooccur):
    cooccur_prob = np.zeros((len(keywords), len(keywords)))
    for i,j in itertools.combinations(range(len(keywords)), 2):
        cooccur_prob[i,j] = np.max([ cooccur[i,j] / float(occur[i]),  cooccur[i,j] / float(occur[j])])
        assert cooccur_prob[i,j] <= 1.0, "%s %s %d %d" % (keywords[i], keywords[j], cooccur[i,j], occur[i], occur[j]) 
        cooccur_prob[j,i] = cooccur_prob[i,j]

    return cooccur_prob


def estimate_cooccur_prob_per_recipe(keywords, occur, cooccur, recipe_no):
    cooccur_prob = np.zeros((len(keywords), len(keywords)))
    for i,j in itertools.combinations(range(len(keywords)), 2):
        cooccur_prob[i,j] = cooccur[i,j] / recipe_no 
        cooccur_prob[j,i] = cooccur_prob[i,j]

    return cooccur_prob


def read_ings(ings_path):
    df = pd.read_csv(ings_path, encoding="utf-8")
    ings = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return ings


def main(params):
    cooccurrence_path = params['cooccurrence_path']
    output_dir = params['output_dir']

    with open(cooccurrence_path, 'rb') as fin:
        keywords, occur, cooccur, recipe_no = pickle.load(fin)
    keywords = np.array(keywords)
    occur = np.array(occur)
    
    #食材以外, 少ないものを除く
    idx = occur > 100
    keywords = keywords[idx]
    occur = occur[idx]
    cooccur = cooccur[idx][:, idx]

    print ("# of keywords:",keywords.size)

    # 共起確率計算
    cooccur_prob = estimate_cooccur_prob(keywords, occur, cooccur)

    # 確率をそのまま距離に つまり共起しないものを同じクラスタにまとめる
    d_array = ssd.squareform(cooccur_prob) # linkageへの入力は(1,2),...(1,n),(2,3)...というベクトル
    coprob_mean = np.mean(d_array[d_array > 0])
    print ("coprob_mean %.3e" % coprob_mean)
    result = sch.linkage(d_array, method = 'complete') 
    
    ths = np.array(result[::-1, 2]) #クラスタ数が少ない順の閾値

    mx_cluster = len(np.where(ths > 0)[0])
    th = coprob_mean 
    n_food = 0
    cls = sch.fcluster(result, th, "distance")
    print ("n_cluster %d" % len(set(cls)))
    for c in set(cls):
        same_cls = keywords[cls == c]
        n = same_cls.size
        idx = np.argsort(occur[cls == c])[::-1] # 降順
        with open(os.path.join(output_dir, "cluster_%04d.txt" % c), "wt") as fout:
            writer = csv.writer(fout, delimiter = '\t')
            writer.writerow(["label", "occur"])
            foods = np.c_[ same_cls[idx].reshape(n,1), occur[cls == c][idx].reshape(n,1)]            
            n_food += len(foods)
            writer.writerows(foods)

    print ("n_food", n_food)

    #plt.figure(figsize=(7,7))
    #plt.plot(range(1, len(ths)+1), ths)
    #plt.xlim(0,mx_cluster)
    #plt.ylim(0,1)
    #plt.savefig(os.path.join(output_dir, "th_for_cluster_no.png"))

    plt.figure(figsize=(7,7))
    plt.plot(range(1, len(ths)+1), ths)
    plt.hlines(th, 0, len(set(cls)), color='r')
    plt.vlines(len(set(cls)), 0, th, color='r')
    plt.plot(len(set(cls)), th, marker='D', ms=10, color='r')
    plt.xlim(0,mx_cluster)
    plt.ylim(0,1)
    plt.savefig(os.path.join(output_dir, "th_for_cluster_no_%.2e_%d.png" % (th, len(set(cls)))))

    with open(os.path.join(output_dir, 'clustering_keywords.pickle'), 'wb') as fout:
        pickle.dump((keywords, occur, cooccur, recipe_no), fout, protocol=0)


if __name__ == '__main__':
    params = parse()
    main(params)
