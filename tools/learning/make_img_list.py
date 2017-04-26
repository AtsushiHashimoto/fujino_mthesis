# _*_ coding: utf-8 -*-

import os 
import csv
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module.ImgList import ImgList 

CLUSTER = "0002"
USE_DIR = "今日の料理,ジャガイモ_625,豆腐_625"
MODE = "center"
OUTPUT_DIR = "/home/fujino/work/hdf5/%s/init_REG3_P1N4"% (CLUSTER)
RECIPE_IMG_DIR_PATH="/home/fujino/work/output/recipe_image_directory.json"
DATA_DIR = "/media/EXT/caffe_db/recipe"
CLUSTER_PATH = "/home/fujino/work/output/clustering/cluster_%s.txt" % CLUSTER
LABEL_DIR = "/home/fujino/work/output/FlowGraph/label/"
SUFFIX = "test"
SEED = 0
STEP_DIR = "/DATA/IMG/step/"
REGION = 3


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_dir', help=u'output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('-rcp_loc_steps_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default=RECIPE_IMG_DIR_PATH)
    parser.add_argument('-recipe_dir', help=u"recipeIDのリスト含むディレクトリ", 
                        default=DATA_DIR)
    parser.add_argument('-use_dirs', help=u"directories in npy directory (dir1,dir2,...)", type=str,
                        default=USE_DIR)
    parser.add_argument('-cluster_path', help=u'cluster_*.txt',
                        default=CLUSTER_PATH)
    parser.add_argument('-label_dir', help=u'output directory of generate label',
                        default=LABEL_DIR)
    parser.add_argument('-seed', help=u"numpy seed", type=int, 
                        default=SEED)
    parser.add_argument('-suffix', help=u"recipe suffix (train, test, valid)", 
                        default=SUFFIX)
    parser.add_argument('-step_dir', help=u'/DATA/IMG/step/',
                        default=STEP_DIR)
    parser.add_argument('-mode', help=u'all:use all regions, center:use a center region, cover:画像をカバーするようになるべく多い領域をとる(アルゴリズムは修論参照)',
                        default=MODE)
    parser.add_argument('-region', help=u"modeがcoverまたはallの場合にどのくらい細かく領域をとるか(Nなら修論の閾値thetaを1/(2^2), 1/(3^2), ... 1/(N^2)というようにN-1個設定する)", type=int, 
                        default=REGION)
    parser.add_argument('-ss_dir', help=u"selective searchの結果を保存するディレクトリ", 
                        default="../exp/selective_search")


    params = parser.parse_args()

    return vars(params)

def load_cluster_labels(cluster_path):
    labels = []
    with open(cluster_path, "rt") as fin:
        reader = csv.reader(fin, delimiter='\t')
        next(reader) #skip header
        for row in reader:
            labels.append(unicode(row[0], 'utf-8'))
    return labels

def main(
    output_dir, rcp_loc_steps_path,
    recipe_dir, use_dirs, cluster_path, label_dir, suffix, seed, step_dir, mode, region, ss_dir):


    with open(rcp_loc_steps_path, 'r') as fin:
        rcp_loc_steps = json.load(fin)

    use_dirs = use_dirs.split(",")
    print "use %s" % " ".join(use_dirs)
    print suffix 
    print mode

    recipe_ids = pd.concat([pd.read_csv(os.path.join(recipe_dir, use_dir, "recipes_%s.tsv" % suffix), 
                                        delimiter='\t', encoding='utf-8', header=None) 
                                    for use_dir in use_dirs])
    recipe_ids = np.unique(recipe_ids.as_matrix())

    print "make img list..."
    img_list = ImgList(seed=seed, ss_dir=ss_dir)
    img_list.make(recipe_ids, rcp_loc_steps, step_dir, label_dir, mode, region)
    cluster_labels = load_cluster_labels(cluster_path)
    img_list.add_label(cluster_labels, label_dir)
    img_list.save(output_dir, suffix)

if __name__ == '__main__':
    params = parse()
    main(**params)


