# _*_ coding: utf-8 -*-

import gc
import os
import sys
import glob
import argparse
import multiprocessing

import caffe
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module.ExecutionTime import ExecutionTime 
from module.ImgList import ImgList 
from module import caffe_module 


CLUSTER = "0015"
DIR_SUF = "_REG3_P1N4_step"
ITER = 7178 

DATA_NO = 100 

HOME = "/home/fujino/"
OUTPUT_DIR = HOME + "work/hdf5/%s/refine%s/iter%d/" % (CLUSTER, DIR_SUF, ITER)
MODEL = "/home/fujino/work/hdf5/%s/init%s/prototxt/deploy.prototxt" % (CLUSTER, DIR_SUF)
PRETRAINED = "/home/fujino/work/hdf5/%s/init%s/snapshot/REG3stp_iter_%d.caffemodel" % (CLUSTER, DIR_SUF, ITER)
MEAN_PATH = "/home/fujino/work/hdf5/%s/init%s/train_mean.npy" % (CLUSTER, DIR_SUF)

#HOME = "/home/fujino/ebisu_home/"
#OUTPUT_DIR = "/home/fujino/output/" 
#MODEL = HOME + "/work/hdf5/%s/init%s/prototxt/deploy.prototxt" % (CLUSTER, DIR_SUF)
#PRE = "REG3stp"
#PRETRAINED = "/home/fujino/work/snapshot/%s_iter_%d.caffemodel" % (CLUSTER, DIR_SUF, PRE, ITER)
#MEAN_PATH = HOME + "/work/hdf5/%s/init%s/train_mean.npy" % (CLUSTER, DIR_SUF)

IMG_LIST_PATHS = [
HOME + "/work/hdf5/%s/init%s/img_list_train%04d.tsv" % (CLUSTER, DIR_SUF, i) for i in range(DATA_NO)]
IMG_LIST_PATHS = [path for path in IMG_LIST_PATHS if os.path.exists(path)]
DATA_PATHS = [
HOME + "/work/hdf5/%s/init%s/data_train%04d.npy" % (CLUSTER, DIR_SUF, i) for i in range(DATA_NO)]
DATA_PATHS = [path for path in DATA_PATHS if os.path.exists(path)]


#REGION = "cover"
#REGION = "center"
#OUTPUT_DIR = "/media/EXT/caffe_db/hdf5/%s/%s/refine%s/iter%d/" % (CLUSTER, REGION, DIR_SUF, ITER)
#MODEL = "/media/EXT/caffe_db/hdf5/%s/%s/init%s/prototxt/deploy.prototxt" % (CLUSTER, REGION, DIR_SUF)
#PRETRAINED = "/media/EXT/caffe_db/hdf5/%s/%s/init%s/snapshot/_iter_%d.caffemodel" % (CLUSTER, REGION, DIR_SUF, ITER)
#MEAN_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/init%s/train_mean.npy" % (CLUSTER, REGION, DIR_SUF)
#
#IMG_LIST_PATHS = [
#"/media/EXT/caffe_db/hdf5/%s/%s/init%s/img_list_train%04d.tsv" % (CLUSTER, REGION, DIR_SUF, i) for i in range(DATA_NO)]
#IMG_LIST_PATHS = [path for path in IMG_LIST_PATHS if os.path.exists(path)]
#DATA_PATHS = [
#"/media/EXT/caffe_db/hdf5/%s/%s/init%s/data_train%04d.npy" % (CLUSTER, REGION, DIR_SUF, i) for i in range(DATA_NO)]
#DATA_PATHS = [path for path in DATA_PATHS if os.path.exists(path)]


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir', help=u'output directory',
                        default = OUTPUT_DIR)
    parser.add_argument('-model_path', help=u'deploy.prototxt',
                        default = MODEL)
    parser.add_argument('-pretrained_path', help=u'pretrained.caffemodel',
                        default = PRETRAINED)
    parser.add_argument('-mean_path', help=u'mean.npy',
                        default = MEAN_PATH)
    parser.add_argument('-db_dir', help=u'databaseを保存したディレクトリ',
                        default = "")
    #parser.add_argument('-img_list_paths', help=u'img_list.tsv', action="append",
    #                    default = IMG_LIST_PATHS)
    #parser.add_argument('-data_paths', help=u'img_list.tsv', action="append",
    #                    default = DATA_PATHS)

    params = parser.parse_args()

    return vars(params)


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, model_path, pretrained_path, mean_path):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_path = model_path
        self.pretrained_path = pretrained_path 
        self.mean_path = mean_path

    def extract_features(self, net, img_list_path, data_path):
        print (img_list_path)
        print (data_path)

        p = os.path.basename(img_list_path)

        print (p, "load...")
        img_list = ImgList().load(img_list_path)
        data = np.load(data_path).astype("f4")

        print (p, "select data...")
        labels = img_list.df.ix[:, "__background__":].ix[:,1:] # __background__以外のラベル [0, 1]  
        idx = np.any(labels == 1, axis=1) # __background__以外のラベルのいずれかが1のindex  

        self.imgs = img_list.df.ix[idx, :7].as_matrix()
        self.labels = labels.ix[idx,:].as_matrix()
        #data = data[idx, :].transpose((0, 2, 3, 1)) # 保存はN, d, x, yの順
        data = data[idx, :]
        print (data.shape)

        print (p, "extract features...")
        out = net.forward_all(**{net.inputs[0]: data, 'blobs': ['fc7']}) #predictを使うと真ん中切り出しなどの余計な処理をする
        self.features = out["fc7"].copy()
        self.probs = out["cls_prob"].copy()
        print (p, "done!")

        del data
        del img_list
        gc.collect()

    def run(self):
        caffe.set_mode_gpu() # プロセスごとに行う必要あり
        caffe.set_device(0) #gpu_id
        net = caffe_module.load_net(self.model_path, self.pretrained_path, self.mean_path)
        while True:
            paths = self.task_queue.get()
            if paths is None:
                self.task_queue.task_done()
                break
            self.extract_features(net, *paths)
            self.task_queue.task_done()
            self.result_queue.put([self.imgs, self.labels, self.features, self.probs])
        return


def get_features(model_path, pretrained_path, mean_path, img_list_paths, data_paths):
    ex_time = ExecutionTime()

    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    num_consumers = np.min([4, len(img_list_paths)]) #1つ830MBだが画像一気に突っ込むと一つのネットワークしか回さないのでロードさせるために8でいい?
    consumers = [Consumer(tasks, results, model_path, pretrained_path, mean_path) for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()

    print ('Creating %d consumers' % num_consumers)
    ex_time.start()
    [tasks.put(paths) for paths in zip(img_list_paths, data_paths)]
    [tasks.put(None) for i in range(num_consumers)]# 終了
    tasks.join() # 全てのタスクの終了を待つ
    result_arr = np.array([results.get() for path in img_list_paths])
    imgs = np.vstack(result_arr[:, 0])
    labels = np.vstack(result_arr[:, 1])
    features = np.vstack(result_arr[:, 2])
    probs = np.vstack(result_arr[:, 3])

    ex_time.end()

    return imgs, labels, features, probs


def main(model_path, pretrained_path, mean_path, db_dir, output_dir):
    os.mkdir(output_dir)

    img_list_path = os.path.join(db_dir, "img_list_train*.tsv") 
    img_list_paths = sorted(glob.glob(img_list_path))
    data_path = os.path.join(db_dir, "data_train*.npy") 
    data_paths = sorted(glob.glob(data_path))

    if len(img_list_paths) == 0 or len(data_paths) == 0:
        print (img_list_paths)
        print (data_paths)
        print ("img_list_paths or data_paths: empty")
        sys.exit()

    print ("get features ...")
    imgs, labels, features, probs = get_features(model_path, pretrained_path, mean_path, img_list_paths, data_paths)

    print (imgs.shape)
    print (imgs)
    print (labels.shape)
    print (labels)
    print (features.shape)
    print (features)
    print (probs.shape)
    print (probs)

    np.save(os.path.join(output_dir, 'f_imgs.npy'), imgs)
    np.save(os.path.join(output_dir, 'f_labels.npy'), labels)
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'f_probs.npy'), probs)


if __name__ == "__main__":
    params = parse()
    main(**params)
