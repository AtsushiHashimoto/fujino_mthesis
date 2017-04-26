# _*_ coding: utf-8 -*-

import os 
import csv
import sys
import json
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module.ExecutionTime import ExecutionTime 
from module.ImgList import ImgList 
from module.ImgList import make_hdf5_db 

#CLUSTER = "0057"
#USE_DIR = "今日の料理,玉葱_625"
#CLUSTER = "0023"
#USE_DIR = "今日の料理,キャベツ_625"
#CLUSTER = "0021"
#USE_DIR = "今日の料理,トマト_625"
#CLUSTER = "0024"
#USE_DIR = "今日の料理,大根_625,ピーマン_625"
#CLUSTER = "0002"
#USE_DIR = "今日の料理,ジャガイモ_625,豆腐_625"
#CLUSTER = "0009"
#USE_DIR = "今日の料理,卵_625"
#CLUSTER = "0015"
#USE_DIR = "neg0015,ニンジン_625"
CLUSTER = "0015"
USE_DIR = "neg0015,ニンジン_625"
#CLUSTER = "potato"
#USE_DIR = "neg0002,ジャガイモ_625"
#MODE = "center"
MODE = "cover"
#OUTPUT_DIR = "/media/EXT/caffe_db/hdf5/%s/%s/init_REG3_P1N4"% (CLUSTER, MODE)
OUTPUT_DIR = "/home/fujino/work/hdf5/%s/init_REG3_P1N4"% (CLUSTER)
RECIPE_IMG_DIR_PATH="/home/fujino/work/output/recipe_image_directory.json"
IMG_SIZE = 224
DATA_DIR = "/media/EXT/caffe_db/recipe"
#DATA_DIR = "/home/fujino/work/output/caffe_db/recipe"
CLUSTER_PATH = "/home/fujino/work/output/clustering/cluster_%s.txt" % CLUSTER
LABEL_DIR = "/home/fujino/work/output/FlowGraph/label/"
SUFFIX = "train"
SEED = 0
STEP_DIR = "/DATA/IMG/step/"
REGION = 3


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_dir', help=u'output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('-rcp_loc_steps_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default=RECIPE_IMG_DIR_PATH)
    parser.add_argument('-img_size', help=u"img size",  type=int,
                        default=IMG_SIZE)
    parser.add_argument('-data_dir', help=u"recipeIDのリスト含むディレクトリ", 
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
    output_dir, rcp_loc_steps_path, img_size,
    data_dir, use_dirs, cluster_path, label_dir, suffix, seed, step_dir, mode, region, ss_dir):


    with open(rcp_loc_steps_path, 'r') as fin:
        rcp_loc_steps = json.load(fin)

    use_dirs = use_dirs.split(",")
    print "use %s" % " ".join(use_dirs)
    print suffix 
    print mode

    recipe_ids = pd.concat([pd.read_csv(os.path.join(data_dir, use_dir, "recipes_%s.tsv" % suffix), 
                                        delimiter='\t', encoding='utf-8', header=None) 
                                    for use_dir in use_dirs])
    recipe_ids = np.unique(recipe_ids.as_matrix())

    print "make img list..."
    img_list = ImgList(seed=seed, ss_dir=ss_dir)
    img_list.make(recipe_ids, rcp_loc_steps, step_dir, label_dir, mode, region)

    MAX_IMG_PER_DB = 5000 # 10000 / 2(反転)
    img_lists = img_list.split(MAX_IMG_PER_DB)

    ex_time = ExecutionTime()

    n_img = 0
    cluster_labels = load_cluster_labels(cluster_path)
    sum_matrix = np.zeros((3, img_size, img_size), dtype='f4')
    for db_no, img_list in enumerate(img_lists):
        print "make data %d/%d..." % (db_no, len(img_lists))
        ex_time.start()
        img_list.add_label(cluster_labels, label_dir)
        img_list.save(output_dir, suffix+"%04d"%db_no)
        X = img_list.make_image_data(img_size, rcp_loc_steps, output_dir, suffix+"%04d"%db_no)
        n_img += len(X)
        sum_matrix += np.sum(X, axis=0)
        ex_time.end()
    mean = (sum_matrix + sum_matrix[:,:,::-1]) / (float(n_img) * 2)
    mean_path = os.path.join(output_dir, "%s_mean.npy"%suffix)
    np.save(mean_path, mean)

    n_db = len(img_lists)
    #n_db = 25

    print "make hdf5..."
    data_paths = [os.path.join(output_dir, 'data_%s.npy' % (suffix+"%04d"%db_no)) for db_no in range(n_db)]
    img_list_paths = [os.path.join(output_dir, 'img_list_%s.tsv' % (suffix+"%04d"%db_no)) for db_no in range(n_db)]
    ex_time.start()
    make_hdf5_db(img_list_paths, data_paths, mean_path, output_dir, suffix)
    ex_time.end()
    print "finish"


if __name__ == '__main__':
    params = parse()
    main(**params)


