# -*- coding: utf-8 -*-

import io
import os
import sys
import csv
import json
import argparse

import pandas as pd
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from frcnn import _init_paths
import caffe

from frcnn.frcnn_module import Frcnn 


#ITER = 35920
#CLUSTER = "0002"
#DIR_SUF = "_REG3_P1N4"

ITER = 35900 
CLUSTER = "0015"
DIR_SUF = "_REG3_P1N4"

OUTPUT_PATH = "/home/fujino/work/hdf5/%s/init%s/frcnn/test_frcnn_result_%d.json" % (CLUSTER, DIR_SUF, ITER)
MODEL = "/home/fujino/work/hdf5/%s/init%s/frcnn/prototxt/test.prototxt" % (CLUSTER, DIR_SUF)
PRETRAINED = "/home/fujino/work/hdf5/%s/init%s/frcnn/snapshot/frcnn_iter_%d.caffemodel" % (CLUSTER, DIR_SUF, ITER)

#OUTPUT_PATH = "/home/fujino/work/hdf5/%s/refine%s/iter7254/frcnn/test_frcnn_result_%d.json" % (CLUSTER, DIR_SUF, ITER)
#MODEL = "/home/fujino/work/hdf5/%s/refine%s/iter7254/frcnn/prototxt/test.prototxt" % (CLUSTER, DIR_SUF)
#PRETRAINED = "/home/fujino/work/hdf5/%s/refine%s/iter7254/frcnn/snapshot/frcnn_iter_%d.caffemodel" % (CLUSTER, DIR_SUF, ITER)

#OUTPUT_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/test_frcnn_result_%d.json" % (CLUSTER, REGION, DIR_SUF, ITER)
#MODEL = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/prototxt/test.prototxt" % (CLUSTER, REGION, DIR_SUF)
#PRETRAINED = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/snapshot/frcnn_iter_%d.caffemodel" % (CLUSTER, REGION, DIR_SUF, ITER)

#OUTPUT_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/test_frcnn2_result_%d.json" % (CLUSTER, REGION, DIR_SUF, ITER)
#MODEL = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/prototxt/test.prototxt" % (CLUSTER, REGION, DIR_SUF)
#PRETRAINED = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/snapshot/frcnn2_iter_%d.caffemodel" % (CLUSTER, REGION, DIR_SUF, ITER)

CLUSTER_PATH = "/home/fujino/work/output/clustering/cluster_%s.txt" % CLUSTER
INPUT_PATHS = [
    #"/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER
    "/home/fujino/work/hdf5/0015/img_list_with_labels_test_annotated.tsv"
    #"/media/EXT/caffe_db/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER
]



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_path', help=u'output path(result.json)',
                        default=OUTPUT_PATH)
    parser.add_argument('-model_path', help=u'deproy.prototxt',
                        default = MODEL)
    parser.add_argument('-pretrained_path', help=u'pretrained.caffemodel',
                        default = PRETRAINED)
    parser.add_argument('-cluster_path', help=u'cluster_XXXX.txt',
                        default = CLUSTER_PATH)
    parser.add_argument('-input_paths', help=u'*.tsv 一行目にpathがあって画像パスが書いているファイル', action="append",
                        default = [])
    parser.add_argument('-step_dir', help=u'step_dir', default=None)

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
    if step_dir != None:
        filename = os.path.basename(img_path)
        dir_no = os.path.basename(os.path.dirname(img_path))
        img_path = os.path.join(step_dir, dir_no, filename)
    return img_path


def predict(input_paths, labels, net, step_dir=None):
    result = {}
    for im_no, input_path in enumerate(input_paths):
        df = pd.read_csv(input_path, delimiter='\t',  encoding='utf-8')
        df = df.fillna("")
        for i, img_path in enumerate((df.ix[:, "path"]).as_matrix()):
            #img_labels = row[7]
            result[img_path] = {}
            if i % 10 == 0:
                print "%d %d/%d %s   \r" % (im_no,i,len(df),img_path),
                sys.stdout.flush()
            if step_dir != None:
                c_img_path = convert_path(step_dir, img_path)
            img = cv2.imread(c_img_path)
            scores, boxes = net.im_detect(img)
            result[img_path]["predicts"] = scores.tolist()
            boxes[:, 2::4] -= boxes[:, 0::4] # w = xmax - xmin
            boxes[:, 3::4] -= boxes[:, 1::4] # h = ymax - ymin
            boxes = boxes.astype(np.int)
            result[img_path]["boxes"] = (boxes.astype(np.int)).tolist() 
    return result


def main(model_path, 
        pretrained_path,
        cluster_path, input_paths, 
        output_path,
        step_dir = None):
    caffe.set_mode_gpu()
    labels = load_cluster_labels(cluster_path)
    labels = ["__background__"] + labels
    net = Frcnn(pretrained_path, model_path, labels, score_thresh=0)
    print pretrained_path
    print " ".join(labels)
    result = predict(input_paths, labels, net, step_dir)
    with io.open(output_path, 'w', encoding='utf-8') as fout:
        data = json.dumps(result, fout, indent=2, sort_keys=True, ensure_ascii=False)
        fout.write(unicode(data)) # auto-decodes data to unicode if str

if __name__ == "__main__":
    params = parse()
    main(**params)
    #for iter in [10000, 20000, 30000, 40000, 50000]:
    #    PRETRAINED = "/home/fujino/work/output/caffe_db/hdf5/%s/update/snapshot/frcnn_imgnet_pretrain_iter_%d.caffemodel" % (CLUSTER,iter) 
    #    params["pretrained_path"] = PRETRAINED
    #    OUTPUT_PATH = "/home/fujino/work/output/caffe_db/hdf5/%s/update/test_frcnn_imgnetpre_result_%d.json" % (CLUSTER, iter)
    #    params["output_path"] = OUTPUT_PATH 
    #    print params
    #    main(**params)
