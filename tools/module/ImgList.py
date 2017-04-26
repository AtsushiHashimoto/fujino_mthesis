# _*_ coding: utf-8 -*-

import gc
import os 
import sys
import json
import h5py
import traceback
import concurrent.futures 

import numpy as np
import pandas as pd
import caffe

from ExecutionTime import ExecutionTime
from ObjectDetection import ObjectDetection


def get_label_path(label_dir, img_path):
    root = os.path.splitext(os.path.basename(img_path))[0]
    img_dir_no = os.path.basename(os.path.dirname(img_path))
    label_path = os.path.join(label_dir, img_dir_no, '%s.json'%root)
    return label_path


def load_labels(input_path, NEtype_list=["F"], use_ontology=True, step=None):
    """
        input_path: generate_labelsの出力結果
        NEtype_list: 使用するNEtypeのリスト
        use_ontology: オントロジーを使う:True 使わない:False
    """
    with open(input_path, 'r') as fin:
        img_label = json.load(fin)

    labels = []
    for node in img_label.values():
        if node["NEtype"] in NEtype_list:
            if step is None or node["step"] == step:
                if use_ontology:
                    if node["ontology"] != None:
                        labels.append(node["ontology"])
                else:
                    labels.append(node["rNE"])

    labels = list(set(labels)) #重複を除去

    return labels


def make_hdf5_db(img_list_paths, data_paths, mean_path, output_dir, prefix):
    output_dir = os.path.abspath(output_dir)
    mean = np.load(mean_path).astype("f4")
    save_files = []
    # split - 1を反転してくっつける
    pX = np.load(data_paths[-1]).astype("f4")
    pX -= mean
    img_list = ImgList().load(img_list_paths[-1])
    pY = img_list.df.ix[:,"__background__":].as_matrix().astype("f4")
    for data_path, img_list_path in zip(data_paths, img_list_paths):
        X = np.load(data_path).astype("f4")
        X -= mean
        img_list = ImgList().load(img_list_path)
        Y =  img_list.df.ix[:,"__background__":].as_matrix().astype("f4")
        # 左右反転
        pX = pX[:,:,:,::-1]
        data = np.r_[X, pX] 
        labels = np.r_[Y, pY] 
        suffix = os.path.splitext(os.path.basename(data_path))[0].split("_")[-1]
        print "\t", suffix
        print "\tshape of data",data.shape
        print "\tshape of labels",labels.shape
        save_file = os.path.join(output_dir, 'db_%s.h5'%(suffix))
        with h5py.File(save_file,'w') as fout: 
            fout.create_dataset( 'data', data=data) 
            fout.create_dataset( 'labels', data=labels)
        save_files.append(save_file)
        pX = X
        pY = Y
        del X
        del Y
        gc.collect()
    with open(os.path.join(output_dir, "%s_dblist.txt" % prefix),'w') as fout:
       for save_file in save_files:
           fout.write("%s\n" % save_file) # list all h5 files you are going to use    
    del pX
    del pY
    gc.collect()


def make_hdf5_db_random_bg(img_list_paths, data_paths, mean_path, output_dir, prefix):
    output_dir = os.path.abspath(output_dir)
    mean = np.load(mean_path).astype("f4")
    save_files = []
    # split - 1を反転してくっつける
    pX = np.load(data_paths[-1]).astype("f4")
    pX -= mean
    img_list = ImgList().load(img_list_paths[-1])
    pY = img_list.df.ix[:,"__background_0__":].as_matrix().astype("f4")
    for data_path, img_list_path in zip(data_paths, img_list_paths):
        X = np.load(data_path).astype("f4")
        X -= mean
        img_list = ImgList().load(img_list_path)
        Y =  img_list.df.ix[:,"__background_0__":].as_matrix().astype("f4")
        # 左右反転
        pX = pX[:,:,:,::-1]
        data = np.r_[X, pX] 
        labels = np.r_[Y, pY] 
        suffix = os.path.splitext(os.path.basename(data_path))[0].split("_")[-1]
        print "\t", suffix
        print "\tshape of data",data.shape
        print "\tshape of labels",labels.shape
        save_file = os.path.join(output_dir, 'db_%s.h5'%(suffix))
        with h5py.File(save_file,'w') as fout: 
            fout.create_dataset( 'data', data=data) 
            fout.create_dataset( 'labels', data=labels)
        save_files.append(save_file)
        pX = X
        pY = Y
        del X
        del Y
        gc.collect()
    with open(os.path.join(output_dir, "%s_dblist.txt" % prefix),'w') as fout:
       for save_file in save_files:
           fout.write("%s\n" % save_file) # list all h5 files you are going to use    
    del pX
    del pY
    gc.collect()


class ImgList:
    def __init__(self, seed = 0, ss_dir = None):
        self.df = None
        np.random.seed(seed)
        self.ss_dir = ss_dir


    def save(self, output_dir, suffix):
        self.df.to_csv(os.path.join(output_dir, 'img_list_%s.tsv' % suffix), sep="\t", encoding="utf-8", index=False)


    def load(self, path):
        self.df = pd.read_csv(path, delimiter="\t", encoding="utf-8")
        self.df.fillna("")
        return self
        

    def set(self, df):
        self.df = df
        return self
        

    def make(self, recipe_ids, rcp_loc_steps, step_dir, label_dir, mode, th):
        ss = ObjectDetection(result_dir = self.ss_dir)

        def _exsit_label(img_path):
            label_path = get_label_path(label_dir, img_path)
            if os.path.exists(label_path):
                return True
            else:
                return False

        def _is_overlap(rect1, rect2):
            x1, y1, w1, h1 = rect1 
            x2, y2, w2, h2 = rect2 
            if x1 <= x2+w2 and x2 <= x1+w1 and y1 <= y2+h2 and y2 <= y1+h1:
                return True
            else:
                return False
        
        def _IoU(rect1, rect2):
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            area1 = w1*h1 
            area2 = w2*h2 
            if _is_overlap(rect1, rect2):
                intsct_xmin = np.max([x1, x2])
                intsct_ymin = np.max([y1, y2])
                intsct_xmax = np.min([x1+w1, x2+w2])
                intsct_ymax = np.min([y1+h1, y2+h2])
                intsct_area = (intsct_xmax - intsct_xmin) * (intsct_ymax - intsct_ymin)
                score = intsct_area / float(area1 + area2 - intsct_area)
            else:
                score = 0
            return score

        def _center_rect(img, rects):
            img_h, img_w, _ = img.shape
            # evaluate by IoU
            scores = []
            ideal_rect= (0.5*(1 - np.sqrt(0.5))*img_w, 0.5*(1 - np.sqrt(0.5))*img_h, 
                         np.sqrt(0.5)*img_w, np.sqrt(0.5)*img_h) #x,y,w,h
            scores = [_IoU(rect, ideal_rect) for rect in rects] 
            # select a bounding box with maximam IoU 
            idxs = np.argsort(scores)
            idxs = idxs[::-1]
            return rects[idxs[0]]

        def _cover_rects(img, rects, max_split=3):
            sub_rects = []
            img_h, img_w, _ = img.shape
            for n in range(2,max_split+1):
                srects = [rect for rect in rects if rect[2]*rect[3] >= float(img_h * img_w)/(n*n)]
                srects = np.array(sorted(srects, key = lambda x : x[2] * x[3])) #面積小さい順にソート
                selected_srects = [] 
                for rect in srects:
                    if len(selected_srects) == 0:
                        selected_srects = np.array([rect])
                    else:
                        overlap = np.apply_along_axis(_is_overlap, 1, selected_srects, rect) #重なりがあるかどうか
                        if not np.any(overlap): #処理済みのものと一切重なりがなければ追加
                            selected_srects = np.vstack([selected_srects, rect])
                if len(sub_rects) == 0: 
                    sub_rects = selected_srects 
                else:
                    sub_rects = np.vstack([sub_rects, selected_srects])
            sub_rects = list(set([tuple(s) for s in sub_rects])) #重複除去
            sub_rects = [list(s) for s in sub_rects]
            return sub_rects

        def _all_rects(img, rects, max_split=5):
            img_h, img_w, _ = img.shape
            rects = [rect for rect in rects if rect[2]*rect[3] >= float(img_h * img_w)/(max_split*max_split)]
            return rects

        def _get_sub_images(img_path):
            try:
                img, _, regions = ss.detect(img_path, img_size=224, scale=300, sigma=0.7, min_size=10) #経験的
                img_h, img_w, _ = img.shape
                rects = [tuple(region["rect"]) for region in regions 
                            if region["rect"][2] > 10 and region["rect"][3] > 10 
                                and region["rect"][2]*region["rect"][3] > img_h*img_w/100]  #縦・横10ピクセル以上 面積1/100以上
                rects = list(set(rects)) #重複除去
                rects = [list(r) for r in rects]
                resize_h, resize_w, _ = img.shape
                if mode == "center":
                    center_rect = _center_rect(img, rects)
                    return [[img_path, resize_w, resize_h] + center_rect]
                elif mode == "cover":
                    rects = _cover_rects(img, rects, max_split=th)
                    return [[img_path, resize_w, resize_h] + r for r in rects]
                elif mode == "all": #全て
                    rects = _all_rects(img, rects, max_split=th)
                    return [[img_path, resize_w, resize_h] + r for r in regions]
                else:
                    print "unknown mode %s" % mode
                    sys.exit()
            except KeyboardInterrupt:
                print traceback.format_exc(sys.exc_info()[2])
                sys.exit()
            except:
                print "except", img_path
                #print traceback.format_exc(sys.exc_info()[2])
                return [["except", 0, 0, 0, 0, 0, 0]]

        def _make_records(recipe_id):
            if recipe_id in rcp_loc_steps: 
                img_dir_no = rcp_loc_steps[recipe_id]["dir"]
                steps = rcp_loc_steps[recipe_id]["steps"]
                img_paths = [os.path.join(step_dir, img_dir_no, u'%s_%s.jpg'%(recipe_id,step)) for step in steps]
                img_paths = [img_path for img_path in img_paths if _exsit_label(img_path)]
                if len(img_paths) > 0:
                    result = [_get_sub_images(img_path) for img_path in img_paths]
                    print recipe_id
                    return result
                else:
                    return [[["except", 0, 0, 0, 0, 0, 0]]]
            else:
                return [[["except", 0, 0, 0, 0, 0, 0]]]


        ex_time = ExecutionTime()
        ex_time.start()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
        futures = [executor.submit(_make_records, recipe_id) for recipe_id in recipe_ids] 
        img_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        # remove exception...
        img_list = [region_list for recipe_list in img_list for step_list in recipe_list for region_list in step_list 
                        if region_list[0] != "except"]
        print "images", len(img_list)
        self.df = pd.DataFrame(img_list)
        self.df.columns = ["path","resize_w","resize_h","x","y","w","h"]
        executor.shutdown()
        ex_time.end()


    def split(self, max_per_list):
        #shuffle
        cols = self.df.columns
        img_list = self.df.as_matrix()
        np.random.shuffle(img_list) #shuffle
        n_img = len(img_list)
        n_split = (n_img / max_per_list) + 1
        n_img_per_list = n_img / n_split

        img_lists = [ImgList(ss_dir=self.ss_dir).set(pd.DataFrame(img_list[n_img_per_list*s : n_img_per_list*(s+1)], columns=cols)) for s in range(n_split)]
        return img_lists


    def add_label(self, cluster_labels_wo_bg, label_dir, stp=False): 
        y = np.zeros((len(self.df), len(cluster_labels_wo_bg) + 1), dtype='f4') #FRCNNに合わせて0はbackground
        labels = [] 
        for i, line in enumerate(self.df.as_matrix()):
            img_path = line[0]
            label_path = get_label_path(label_dir, img_path)

            if stp:
                s = int((os.path.splitext(os.path.basename(img_path))[0]).split("_")[-1])
            else:
                s = None 
            img_labels = load_labels(label_path, step=s)

            labels.append(";".join(img_labels))
            exist_label = False
            for img_label in img_labels:
                if img_label in cluster_labels_wo_bg:
                    exist_label = True 
                    idx = cluster_labels_wo_bg.index(img_label) + 1 #0はbackground
                    y[i][idx] = 1.0 
            if not exist_label: # background_class
                y[i][0] = 1.0 
    
        self.df["labels"] = labels
        for i, cl in enumerate(["__background__"] + cluster_labels_wo_bg):
            self.df[cl] = y[:, i] 


    def add_label_random_bg(self, cluster_labels_wo_bg, label_dir, n_bg): 
        y = np.zeros((len(self.df), len(cluster_labels_wo_bg) + n_bg), dtype='f4') #FRCNNに合わせて0はbackground
        labels = [] 
        for i, line in enumerate(self.df.as_matrix()):
            img_path = line[0]
            label_path = get_label_path(label_dir, img_path)
            img_labels = load_labels(label_path)
            labels.append(";".join(img_labels))
            exist_label = False
            for img_label in img_labels:
                if img_label in cluster_labels_wo_bg:
                    exist_label = True 
                    idx = cluster_labels_wo_bg.index(img_label) + n_bg #前半n_bgクラスはbackground
                    y[i][idx] = 1.0 
            if not exist_label: # background_class
                idx = np.random.choice(range(n_bg))
                y[i][idx] = 1.0 
    
        self.df["labels"] = labels
        for i, cl in enumerate(["__background_%d__" % bg_no for bg_no in range(n_bg)] + cluster_labels_wo_bg):
            self.df[cl] = y[:, i] 


    def make_image_data(self, img_size, rcp_loc_steps, output_dir, suffix): 
        def _set_img(X, index, line):
            img_path, resize_w, resize_h, x, y, w, h = line[:7]
            img = caffe.io.load_image(img_path, color=True)
            img_h, img_w, _ = img.shape
            x = int(float(img_w)/resize_w * x)
            y = int(float(img_h)/resize_h * y)
            w = int(float(img_w)/resize_w * w)
            h = int(float(img_h)/resize_h * h)
            crop_img = img[y:y+h,x:x+w]
            crop_img = caffe.io.resize(crop_img, (img_size, img_size, 3)) # resize to fixed size
            X[index] = crop_img.transpose((2, 0, 1))

        X = np.zeros( (len(self.df), 3, img_size, img_size), dtype='f4' ) 
        [_set_img(X,i,line) for i, line in enumerate(self.df.as_matrix())]
        np.save(os.path.join(output_dir, 'data_%s.npy' % suffix), X)
        return X
