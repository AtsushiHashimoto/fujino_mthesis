# -*- coding: utf-8 -*-

import os
import io
import sys
import json
import argparse
import multiprocessing
from itertools import chain

import caffe
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module import caffe_module
from module import image
from module.ObjectDetection import ObjectDetection
from module.ExecutionTime import ExecutionTime


CLUSTER = "0015"
USE_DIR = "今日の料理,ニンジン"
#ITER = 7178 
ITER = 0 

#CLUSTER = "0002"
#USE_DIR = "今日の料理,ジャガイモ,豆腐"
#ITER = 4199 * 7 
#REF_ITE=3591*7

DIR_SUF = "_REG3_P1N4"

#OUTPUT_PATH = "/home/fujino/work/hdf5/%s/init%s/test_result_%d.json" % (CLUSTER, DIR_SUF, ITER)
#MODEL = "/home/fujino/work/hdf5/%s/init%s/prototxt/deploy.prototxt" % (CLUSTER, DIR_SUF)
#PRETRAINED = "/home/fujino/work/hdf5/%s/init%s/snapshot/REG3_iter_%d.caffemodel" % (CLUSTER, DIR_SUF, ITER)
#MEAN_PATH = "/home/fujino/work/hdf5/%s/init%s/train_mean.npy" % (CLUSTER, DIR_SUF)

##refine用
#REF_ITE=0 
#OUTPUT_PATH = "/home/fujino/work/hdf5/%s/refine%s/iter%d/test_result_%d.json" % (CLUSTER, DIR_SUF, ITER, REF_ITE)
#MODEL = "/home/fujino/work/hdf5/%s/refine%s/iter%d/prototxt/deploy.prototxt" % (CLUSTER, DIR_SUF, ITER)
#PRETRAINED = "/home/fujino/work/hdf5/%s/refine%s/iter%d/snapshot/ref_iter_%d.caffemodel" % (CLUSTER, DIR_SUF, ITER, REF_ITE)
#MEAN_PATH = "/home/fujino/work/hdf5/%s/init%s/train_mean.npy" % (CLUSTER, DIR_SUF)

# 初期値
REF_ITE=0 
OUTPUT_PATH = "/home/fujino/work/hdf5/%s/refine%s/iter%d/test_result_%d.json" % (CLUSTER, DIR_SUF, ITER, REF_ITE)
MODEL = '/home/fujino/work/data/caffe/VGG_ILSVRC_16_layers_deploy.prototxt'
PRETRAINED = '/home/fujino/work/data/caffe/VGG_ILSVRC_16_layers.caffemodel'
MEAN_PATH = "/home/fujino/work/hdf5/%s/init%s/train_mean.npy" % (CLUSTER, DIR_SUF)

#REGION = "cover"
#OUTPUT_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/init%s/test_result_%d.json" % (CLUSTER, REGION, DIR_SUF, ITER)
#MODEL = "/media/EXT/caffe_db/hdf5/%s/%s/init%s/prototxt/deploy.prototxt" % (CLUSTER, REGION, DIR_SUF)
#PRETRAINED = "/media/EXT/caffe_db/hdf5/%s/%s/init%s/snapshot/_iter_%d.caffemodel" % (CLUSTER, REGION, DIR_SUF, ITER)
#MEAN_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/init%s/train_mean.npy" % (CLUSTER, REGION, DIR_SUF)

RECIPE_IMG_DIR_PATH="/home/fujino/work/output/recipe_image_directory.json"
STEP_DIR = "/DATA/IMG/step/"
DATA_DIR = "/media/EXT/caffe_db/recipe"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_path', help=u'output path(result.json)',
                        default=OUTPUT_PATH)
    parser.add_argument('-model_path', help=u'deploy.prototxt',
                        default = MODEL)
    parser.add_argument('-pretrained_path', help=u'pretrained.caffemodel',
                        default = PRETRAINED)
    parser.add_argument('-mean_path', help=u'mean.npy',
                        default = MEAN_PATH)
    parser.add_argument('-step_dir', help=u'/DATA/IMG/step/',
                        default=STEP_DIR)
    parser.add_argument('-input_path', help=u'*.tsv 一行目にpathがあって画像パスが書いているファイル',
                        default = None)
    parser.add_argument('-ss_dir', help=u"selective searchの結果を保存するディレクトリ", 
                        default="../exp/selective_search")
    #parser.add_argument('-recipe_dir', help=u"data directory", 
    #                    default=DATA_DIR)
    #parser.add_argument('-use_dirs', help=u"directories in npy directory (dir1,dir2,...)", type=str,
    #                    default=USE_DIR)
    #parser.add_argument('-rcp_loc_steps_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
    #                    default=RECIPE_IMG_DIR_PATH)


    params = parser.parse_args()

    return vars(params)


def convert_path(step_dir, img_path):
    if step_dir != None:
        filename = os.path.basename(img_path)
        dir_no = os.path.basename(os.path.dirname(img_path))
        img_path = os.path.join(step_dir, dir_no, filename)
    return img_path


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, model_path, pretrained_path, mean_path, ss, step_dir):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_path = model_path 
        self.pretrained_path = pretrained_path
        self.mean_path = mean_path
        self.ss = ss 
        self.step_dir = step_dir 

    def predict(self, key_img_path, net):
        IMG_SIZE = 224
        result = {}
    
        if self.step_dir != None:
            img_path = convert_path(self.step_dir, key_img_path)
        else:
            img_path = key_img_path
    
        img, _, regions = self.ss.detect(img_path)
        resize_h, resize_w, _ = img.shape
    
        img = caffe.io.load_image(img_path)
        img_h, img_w, _ = img.shape
        boxes = [image.convert_box(region["rect"], (resize_w, resize_h), (img_w, img_h)) for region in regions]
        boxes = [[0,0,img_w,img_h]] + boxes #元の画像も一番最初に入れておく
        boxes = [tuple(b) for b in boxes if b[2]*b[3] >= float(img_h * img_w) / 25.0]
        boxes = list(set(boxes)) # 重複除去
        boxes = sorted(boxes, key = lambda x : x[2] * x[3], reverse=True) # 面積順に並び替え
        imgs = np.array([caffe.io.resize(img[y:y+h,x:x+w], (IMG_SIZE, IMG_SIZE, 3)).transpose((2, 0, 1)) for x,y,w,h in boxes])
        #imgs = imgs.transpose((0, 3, 1, 2)) # 保存はN, d, x, yの順
        #prediction = net.predict(imgs)
        out = net.forward_all(**{net.inputs[0]: imgs}) #predictを使うと真ん中切り出しなどの処理をする
        prediction = out["cls_prob"].copy()
        n_class = len(prediction[0])
        result["boxes"] = [box * n_class for box in boxes] #FRCNNにあわせてクラスごとに矩形領域を保存(クラスに関係なく同じ領域)
        result["predicts"] = prediction.tolist()
        print key_img_path, len(prediction)
        return (key_img_path, result)

    def run(self):
        caffe.set_mode_gpu() # プロセスごとに行う必要あり
        caffe.set_device(0) #gpu_id
        net = caffe_module.load_net(self.model_path, self.pretrained_path, self.mean_path)
        while True:
            path = self.task_queue.get()
            if path is None:
                self.task_queue.task_done()
                break
            prediction = self.predict(path, net)
            self.task_queue.task_done()
            self.result_queue.put(prediction)
        return


def make_paths(recipe_id, rcp_loc_steps, step_dir):
    if recipe_id in rcp_loc_steps:
        img_dir_no = rcp_loc_steps[recipe_id]["dir"]
        steps = rcp_loc_steps[recipe_id]["steps"]
        paths = [os.path.join(step_dir, img_dir_no, u'%s_%s.jpg'%(recipe_id,step)) for step in steps]
        return paths
    else:
        return None


def main(model_path, 
        pretrained_path, mean_path,
        input_path,
        output_path,
        step_dir,
        ss_dir):
    ss = ObjectDetection(result_dir = ss_dir)
    ex_time = ExecutionTime()

    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    #入力する画像パス
    df = pd.read_csv(input_path, delimiter='\t',  encoding='utf-8')
    df = df.fillna("")
    paths = df.ix[:, "path"].as_matrix()
    paths = [path for path in paths if os.path.exists(path)] 

    print "predict"
    ex_time.start()
    num_consumers = 4 #経験的
    print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results, model_path, pretrained_path, mean_path, ss, step_dir) for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    [tasks.put(path) for path in paths]
    [tasks.put(None) for i in range(num_consumers)]# 終了
    tasks.join() # 全てのタスクの終了を待つ
    result= [results.get() for path in paths]
    ex_time.end()

    result = dict(result)

    print "save"
    ex_time.start()
    with io.open(output_path, 'w', encoding='utf-8') as fout:
        data = json.dumps(result, fout, indent=2, sort_keys=True, ensure_ascii=False)
        fout.write(unicode(data)) # auto-decodes data to unicode if str
    ex_time.end()


if __name__ == "__main__":
    params = parse()
    main(**params)
