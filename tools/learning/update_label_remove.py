# _*_ coding: utf-8 -*-

import os
import sys
import glob
import argparse
from itertools import chain

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module.ImgList import ImgList 
from module.ExecutionTime import ExecutionTime

#CLUSTER = "0015"
#DIR_SUF = "_REG3_P1N4_step"
#ITER = 7178

CLUSTER = "potato"
DIR_SUF = "_REG3_P1N4"
ITER = 7254

OUTPUT_DIR = "/home/fujino/work/hdf5/%s/refine%s/iter%d" % (CLUSTER, DIR_SUF, ITER)
F_IMGS_PATH = "/home/fujino/work/hdf5/%s/refine%s/iter%d/f_imgs.npy" % (CLUSTER, DIR_SUF, ITER)
IMG_LIST_PATHS = [
"/home/fujino/work/hdf5/%s/init%s/img_list_train%04d.tsv" % (CLUSTER, DIR_SUF, i) for i in range(100)]
IMG_LIST_PATHS = [p for p in IMG_LIST_PATHS if os.path.exists(p)]
OUT_IDX_PATHS = [
"/home/fujino/work/hdf5/%s/refine%s/iter%d/fp_estimation/food%03d/out_%d.npy" % (CLUSTER, DIR_SUF, ITER, i, i) for i in range(100)]
OUT_IDX_PATHS = [p for p in OUT_IDX_PATHS if os.path.exists(p)]

#REGION = "cover"
#OUTPUT_DIR = "/media/EXT/caffe_db/hdf5/%s/%s/refine%s/5000" % (CLUSTER, REGION, DIR_SUF)
#F_IMGS_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/refine%s/5000/f_imgs.npy" % (CLUSTER, REGION, DIR_SUF)
#IMG_LIST_PATHS = [
#"/media/EXT/caffe_db/hdf5/%s/%s/init%s/img_list_train%04d.tsv" % (CLUSTER, REGION, DIR_SUF, i) for i in range(100)]
#IMG_LIST_PATHS = [p for p in IMG_LIST_PATHS if os.path.exists(p)]
#OUT_IDX_PATHS = [
#"/media/EXT/caffe_db/hdf5/%s/%s/refine%s/5000/fp_estimation/food%03d/out_%d.npy" % (CLUSTER, REGION, DIR_SUF, i, i) for i in range(100)]
#OUT_IDX_PATHS = [p for p in OUT_IDX_PATHS if os.path.exists(p)]


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir', help=u'output directory',
                        default = OUTPUT_DIR)
    parser.add_argument('-f_imgs_path', help=u'f_img.npy',
                        default = F_IMGS_PATH)
    #parser.add_argument('-img_list_paths', help=u'img_list.tsv', action="append",
    #                    default = IMG_LIST_PATHS)
    parser.add_argument('-db_dir', help=u'databaseを保存したディレクトリ',
                        default = "")
    #parser.add_argument('-out_idx_paths', help=u'img_list.tsv', action="append",
    #                    default = OUT_IDX_PATHS)
    parser.add_argument('-fp_est_dir', help=u'訓練データ選択結果のディレクトリ',
                        default = "")

    params = parser.parse_args()

    return vars(params)


def update_labels(img_list_path, fp_imgs, fp_idxs, output_dir):
    """
    img_list_path:
    fp_img_paths: 
    fp_idxs: [class1_fp_list, class2_fp_list, ... ] 
    output_dir:
    """
    img_list = ImgList()
    img_list.load(img_list_path)

    labels = list(img_list.df.ix[:, "__background__":])[1:]

    all_fp_idxs = list(set(chain.from_iterable(fp_idxs)))
    update = []
    del_idxs = []

    df_imgs = (img_list.df.ix[:, :fp_imgs.shape[1]]).as_matrix()
    for idx in all_fp_idxs:
        fp_img = fp_imgs[idx]
        df_idx = np.where(np.all(df_imgs==fp_img,axis=1))[0]
        if len(df_idx) > 0:
            df_idx = df_idx[0]
            row = img_list.df.ix[df_idx, :] 
            update.append(list(row))

            #該当食材クラスの中でどれが除去されたか(複数の場合もあり)
            remove_labels = []
            for food_class, l in enumerate(labels):
                if idx in fp_idxs[food_class]:
                    remove_labels.append(l)

            #ラベル除去
            orig_labels = row.ix["labels"] 
            orig_labels = orig_labels.split(";")
            new_labels = []
            for l in orig_labels:
                if l not in remove_labels: #前のラベルを残す
                    new_labels.append(l)
                else: #次のラベルから除去
                    img_list.df.ix[df_idx, l] = 0.0 
            if len(new_labels) == 0:
                new_labels = ""
            else:
                new_labels = ";".join(new_labels)
            img_list.df.ix[df_idx, "labels"] = new_labels

            #ラベルが全て消去されたら背景クラスを1にする
            if np.all(img_list.df.ix[df_idx, "__background__":].as_matrix() == 0):
                del_idxs.append(df_idx)
            #    img_list.df.ix[df_idx, "__background__"] = 1.0

    img_list.df = img_list.df.drop(del_idxs)
    suffix = os.path.splitext(os.path.basename(img_list_path))[0].split("_")[-1]
    img_list.save(output_dir, suffix+"ur")

    print "update", len(update)
    if len(update) > 0:
        save_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_list_path))[0] + "ur_imgs.tsv")
        df = pd.DataFrame(update)
        df.to_csv(save_path, index=False, sep="\t", encoding="utf-8", header=list(img_list.df.columns))


def main(f_imgs_path, db_dir, fp_est_dir, output_dir):
    img_paths = np.load(f_imgs_path)

    img_list_path = os.path.join(db_dir, "img_list_train*.tsv") 
    img_list_paths = sorted(glob.glob(img_list_path))
    out_idx_path = os.path.join(fp_est_dir, "food*", "out_*.npy") 
    out_idx_paths = sorted(glob.glob(out_idx_path))

    fp = [list(np.load(path)) for path in out_idx_paths]
    print ("update labels ...")
    ex_time = ExecutionTime()
    ex_time.start()
    for img_list_path in img_list_paths:
        print img_list_path
        update_labels(img_list_path, img_paths, fp, output_dir)
    ex_time.end()


if __name__ == "__main__":
    params = parse()
    main(**params)
