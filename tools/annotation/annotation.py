# -*- coding: utf-8 -*-

import os 
import io
import csv
import sys
import argparse

import cv2
#import caffe
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module import image

CLUSTER = "0002"

#CLUSTER_PATH = "/home/fujino/work/output/clustering/cluster_%s.txt" % CLUSTER
CLUSTER_PATH = "/Users/fujino/Desktop/result/cluster/cluster_%s.txt" % CLUSTER
#IMG_LIST_PATH = "/home/fujino/work/output/caffe_db/hdf5/%s/img_list_with_labels_test.tsv" % CLUSTER
#IMG_LIST_PATH = "/Users/fujino/Desktop/result/%s/cover_P1N4/img_list_train0001.tsv" % CLUSTER
#IMG_LIST_PATH = "/Users/fujino/Desktop/result/0015/new/img_list_train0000.tsv"
IMG_LIST_PATH = "/Users/fujino/Desktop/result/0002/new/img_list_train0000.tsv"
#IMG_LIST_PATH = "/Users/fujino/Desktop/yabai/img_list_train0001.tsv"
STEP_DIR = "/Volumes/COOKPADstep/step" 


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir', help=u'output_dir', default="")
    parser.add_argument('-cluster_path', help=u'cluster_XXXX.txt',
                        default = CLUSTER_PATH)
    parser.add_argument('-img_list_path', help=u'img_list.tsv',
                        default = IMG_LIST_PATH)
    parser.add_argument('-skip_bg', help=u'背景クラスのレシピへのアノテーションを飛ばす?', action="store_true", default=False)
    parser.add_argument('-step_dir', "-s", help=u'step_dir', default=STEP_DIR)

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


def convert_path(step_dir, img_path):
    filename = os.path.basename(img_path)
    dir_no = os.path.basename(os.path.dirname(img_path))
    img_path = os.path.join(step_dir, dir_no, filename)
    return img_path


def resize(img, size):
    img_h, img_w, _ = img.shape
    shp = [img_h, img_w]
    max_edge = np.argmax(shp)
    if shp[max_edge] > size:
        another = (max_edge + 1) % 2
        shp[another] = int(shp[another] * float(size) / shp[max_edge])
        shp[max_edge] = size 
        img = cv2.resize(img,tuple(shp[::-1]))
    return img


def annotation(output_dir, img_list_path, labels, skip_bg, step_dir = None):
    # restart
    output_path = os.path.join(output_dir, os.path.basename(os.path.splitext(img_list_path)[0]) + "_annotation.tsv")
    if os.path.exists(output_path):
        out_df = pd.read_csv(output_path, delimiter='\t',  encoding='utf-8')
        finished = out_df.ix[:, "path"]
        restart = True
    else:
        finished = [] 
        restart = False 
    finished = list(finished)

    in_df = pd.read_csv(img_list_path, delimiter='\t',  encoding='utf-8')
    in_df = in_df.fillna("")

    # 途中経過の食材の偏りをなくすため食材ごとに入力を分類
    # 後で食材1のレシピ-> 食材2のレシピ -> ... 食材nのレシピ -> 食材1のレシピという順でアノテーション
    if skip_bg: #背景クラス含まない
        n_class = len(labels)
    else: #背景クラス含む
        n_class = len(labels) + 1
    dfs = [ [] for l in range(n_class)]
    idxs = []
    label_no = 0 
    pre_recipe_id = None
    for i, row in enumerate(in_df.values.tolist()):
        img_path = row[0]
        recipe_id = os.path.basename(img_path).split("_")[0]
        if pre_recipe_id != recipe_id: #レシピIDが変わったらindexを登録
            if (not skip_bg or label_no != 0) and len(idxs) > 0:
                if skip_bg:
                    dfs[label_no-1].append(idxs)
                else:
                    dfs[label_no].append(idxs)
            idxs = []
            label_no = 0 
        idxs.append(i)
        #レシピごとにどの食材のレシピか決める
        if label_no == 0: # __background__クラス
            for j, l in enumerate(labels): 
                if l in row[7]:
                    label_no = j + 1 #一番最初に出てきた食材のレシピとして登録
                    break
        pre_recipe_id = recipe_id
    if (not skip_bg or label_no != 0) and len(idxs) > 0:
        if skip_bg:
            dfs[label_no-1].append(idxs)
        else:
            dfs[label_no].append(idxs)
    print [len(d) for d in dfs]

    # annotation 
    df_no = 0
    recipe_no = 0 
    with io.open(output_path, "a", encoding='utf-8') as fout:
        if not restart:
            header = "path\tresize_w\tresize_h\tx\ty\tw\th\tlabels\t"
            header += "__background__\t"
            header += "\t".join(labels)
            header += "\n"
            fout.write(header)
        complete = False
        while not complete:
            if recipe_no < len(dfs[df_no]):
                for i in dfs[df_no][recipe_no]:
                    row = in_df.ix[i, :8] # path, resize_w, resize_h, x, y, w, h, labels
                    img_path = row[0]
                    if img_path not in finished: #終わってれば飛ばす
                        img_labels = row[7]
                        #if u"ニンジン" not in img_labels:
                        #    continue
                        print i, img_path
                        print img_labels
                        if step_dir !=None:
                            img_path =  convert_path(step_dir, img_path)
                        #img = caffe.io.load_image(img_path)
                        #plt.imshow(img)
                        #plt.show()
                        #plt.draw()
                        #plt.pause(0.1)
                        img = cv2.imread(img_path)
                        resize_w, resize_h, x, y, w, h = row[1:7]
                        box = image.convert_box([x,y,w,h], [resize_w,resize_h], img.shape[:2][::-1])
                        img = image.rectangle(img, [box])
                        img = resize(img, 400)
                        cv2.imshow("img",img)
                        vec = [0]
                        for label in labels:
                            print "%s? (y) (n)" % label
                            while 1:
                                input = raw_input('>>> ')
                                if input == 'y' or input == 'Y':
                                    vec.append(1)
                                    break
                                elif input == 'n' or input == 'N':
                                    vec.append(0)
                                    break
                                else:
                                    print "Input (y) or (n)."
                        if np.all(np.array(vec) == 0):
                            vec[0] = 1
                            
                        row = [unicode(r) for r in row]
                        row += [unicode(v) for v in vec]
                        write_line = "\t".join(row) + "\n"
                        fout.write(write_line)
                        cv2.destroyAllWindows()
            df_no = (df_no + 1) % len(dfs) #食材の切り替え
            if df_no == 0: #食材が一周したら追加
                recipe_no += 1
            complete = True 
            for d in range(len(dfs)):
                if len(dfs[d]) > recipe_no:
                    complete = False

        cv2.destroyAllWindows()


def main(output_dir, cluster_path, img_list_path, skip_bg, step_dir = None):
    labels = load_cluster_labels(cluster_path)
    print " ".join(labels)
    annotation(output_dir, img_list_path, labels, skip_bg, step_dir)

if __name__ == '__main__':
    params = parse()
    main(**params)
