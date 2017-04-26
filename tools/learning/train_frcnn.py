# -*- coding: utf-8 -*-
#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from frcnn import _init_paths

from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
import caffe
import argparse
import numpy as np

from frcnn import food_recog_db


CLUSTER = "0015"
DIR_SUF = "_REG3_P1N4"

GPU_ID = 0
NAME ="food_recog" 

#IMG_LIST_PATHS = [
#"/home/fujino/work/hdf5/%s/refine%s/iter5000/img_list_train%04du.tsv" % (CLUSTER, DIR_SUF, i) for i in range(100)]
#IMG_LIST_PATHS = [p for p in IMG_LIST_PATHS if os.path.exists(p)]
#OUTPUT_DIR = "/home/fujino/work/hdf5/%s/frcnn%s/snapshot" % (CLUSTER, DIR_SUF)
#SOLVER_PATH = "/home/fujino/work/hdf5/%s/frcnn%s/prototxt/solver.prototxt" % (CLUSTER, DIR_SUF)
#PRETRAINED_PATH = '/home/fujino/work/data/caffe/VGG_ILSVRC_16_layers.caffemodel'

#IMG_LIST_PATHS = [
#"/home/fujino/work/hdf5/%s/refine%s/iter7254/img_list_train%04dur.tsv" % (CLUSTER, DIR_SUF, i) for i in range(100)]
#IMG_LIST_PATHS = [p for p in IMG_LIST_PATHS if os.path.exists(p)]
#OUTPUT_DIR = "/home/fujino/work/hdf5/%s/refine%s/iter7254/frcnn/snapshot" % (CLUSTER, DIR_SUF)
#SOLVER_PATH = "/home/fujino/work/hdf5/%s/refine%s/iter7254/frcnn/prototxt/solver.prototxt" % (CLUSTER, DIR_SUF)

IMG_LIST_PATHS = [
"/home/fujino/work/hdf5/%s/init%s/img_list_train%04d.tsv" % (CLUSTER, DIR_SUF, i) for i in range(100)]
IMG_LIST_PATHS = [p for p in IMG_LIST_PATHS if os.path.exists(p)]
OUTPUT_DIR = "/home/fujino/work/hdf5/%s/init%s/frcnn/snapshot" % (CLUSTER, DIR_SUF)
SOLVER_PATH = "/home/fujino/work/hdf5/%s/init%s/frcnn/prototxt/solver.prototxt" % (CLUSTER, DIR_SUF)

PRETRAINED_PATH = '/home/fujino/work/data/caffe/VGG_ILSVRC_16_layers.caffemodel'
#PRETRAINED_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/snapshot/frcnn_iter_50000.caffemodel" % (CLUSTER, REGION, DIR_SUF)
MAX_ITERS = 35900  #1epochは恐らく画像枚数くらい 画像間でバッチは作れない
SNAPSHOT_ITERS = 3590 

#CLUSTER = "0015"
#CLUSTER = "0002"
#REGION = "cover"
#DIR_SUF = "_REG3_P1N4"
#REGION = "cover"
#IMG_LIST_PATHS = [
#"/media/EXT/caffe_db/hdf5/%s/%s/refine%s/5000/img_list_train%04du.tsv" % (CLUSTER, REGION, DIR_SUF, i) for i in range(100)]
#IMG_LIST_PATHS = [p for p in IMG_LIST_PATHS if os.path.exists(p)]
#OUTPUT_DIR = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/snapshot" % (CLUSTER, REGION, DIR_SUF)
#SOLVER_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/prototxt/solver.prototxt" % (CLUSTER, REGION, DIR_SUF)
#PRETRAINED_PATH = '/home/fujino/work/data/caffe/VGG_ILSVRC_16_layers.caffemodel'
##PRETRAINED_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/snapshot/frcnn_iter_50000.caffemodel" % (CLUSTER, REGION, DIR_SUF)
#MAX_ITERS = 100000  #1epochは恐らく画像枚数くらい 画像間でバッチは作れない
#SNAPSHOT_ITERS = 5000

def parse():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', help='GPU device id to use [0]', type=int,
                        default=GPU_ID) 
    parser.add_argument('-name', help=u'db name', type=str,
                        default = NAME)
    #parser.add_argument('-img_list_paths', help=u'img_list.tsv', action="append",
    #                    default = IMG_LIST_PATHS)
    parser.add_argument('-db_dir', help=u'databaseを保存したディレクトリ',
                        default = "")
    parser.add_argument('-output_dir', help=u'output directory', type=str,
                        default = OUTPUT_DIR)
    parser.add_argument('-solver_path', help='solver.prototxt', type=str,
                        default=SOLVER_PATH)
    parser.add_argument('-pretrained_path', help='pretrained.caffemodel', type=str,
                        default=PRETRAINED_PATH)
    parser.add_argument('-max_iters', help='number of iterations to train', type=int,
                        default=MAX_ITERS)
    parser.add_argument('-snapshot_iters', help='number of iterations to save snapshots', type=int,
                        default=SNAPSHOT_ITERS)
    parser.add_argument('-seed', help='random seed of numpy and frcnn', type=int,
                        default=0)
    

    params = parser.parse_args()
    return vars(params) 


def main(gpu_id, name, db_dir, output_dir,
        solver_path, pretrained_path, max_iters, snapshot_iters, seed):

    img_list_path = os.path.join(db_dir, "img_list_train*.tsv") 
    img_list_paths = sorted(glob.glob(img_list_path))

    if not os.path.exists(output_dir):
        print output_dir , "is not found"
        sys.exit() 

    # set up caffe
    caffe.set_mode_gpu()
    cfg.GPU_ID = gpu_id
    caffe.set_device(gpu_id)

    # set frcnn
    cfg.TRAIN.SNAPSHOT_ITERS = snapshot_iters 

    cfg.RNG_SEED = seed 
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)

    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.IMS_PER_BATCH = 1 
    #cfg.TRAIN.IMS_PER_BATCH = 20 
    cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    cfg.TRAIN.RPN_BATCHSIZE = 256
    cfg.TRAIN.PROPOSAL_METHOD = "gt"
    cfg.TRAIN.BG_THRESH_LO = 0.0
    #cfg.TRAIN.ASPECT_GROUPING = False
    #cfg.TRAIN.ASPECT_GROUPING = True 

    # load imdb
    print "load imdb..."
    imdb = food_recog_db.food_recog(name, img_list_paths, seed=seed) 
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)

    roidb = get_training_roidb(imdb) #flipped

    print 'Output will be saved to `{:s}`'.format(output_dir)
    train_net(solver_path, roidb, output_dir,
              pretrained_model=pretrained_path,
              max_iters=max_iters)

if __name__ == '__main__':
    params = parse()
    main(**params)

