# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
from PIL import Image
import pandas as pd


def append_roi_2_db(roidb, img_file, cls, xmin, ymin, xmax, ymax):
    if img_file not in roidb:
        roidb[img_file] = {}

    if 'boxes' not in roidb[img_file]:
        roidb[img_file]['boxes'] = np.array([[xmin, ymin, xmax, ymax]])
    else:
        roidb[img_file]['boxes'] = np.r_[roidb[img_file]['boxes'], [[xmin, ymin, xmax, ymax]]]

    cls[0] = 0 # __background__ クラスは必ず0にする(py-faster-rcnnの仕様)
    if 'gt_overlaps' not in roidb[img_file]:
        roidb[img_file]['gt_overlaps'] = np.atleast_2d(cls)
    else:
        roidb[img_file]['gt_overlaps'] = np.r_[roidb[img_file]['gt_overlaps'], np.atleast_2d(cls)]

    if 'flipped' not in roidb[img_file]:
        roidb[img_file]['flipped'] = False 

    if 'seg_areas' not in roidb[img_file]:
        roidb[img_file]['seg_areas'] = []
    roidb[img_file]['seg_areas'].append((xmax - xmin) * (ymax - ymin))

    #for cls_idx, v in enumerate(cls):
    #    if v == 1:
    #        if 'boxes' not in roidb[img_file]:
    #            roidb[img_file]['boxes'] = np.array([[xmin, ymin, xmax, ymax]])
    #        else:
    #            roidb[img_file]['boxes'] = np.r_[roidb[img_file]['boxes'], [[xmin, ymin, xmax, ymax]]]

    #        #if 'gt_classes' not in roidb[img_file]:
    #        #    roidb[img_file]['gt_classes'] = []
    #        #roidb[img_file]['gt_classes'].append(cls_idx)

    #        overlap = 1.0
    #        if cls_idx == 0:
    #            overlap = 0
    #        if 'gt_overlaps' not in roidb[img_file]:
    #            roidb[img_file]['gt_overlaps'] = np.zeros((1, len(cls)))
    #            roidb[img_file]['gt_overlaps'][0, cls_idx] = overlap 
    #        else:
    #            roidb[img_file]['gt_overlaps'] = np.r_[roidb[img_file]['gt_overlaps'], np.zeros((1, len(cls)))]
    #            roidb[img_file]['gt_overlaps'][-1, cls_idx] = overlap 

    #        if 'flipped' not in roidb[img_file]:
    #            roidb[img_file]['flipped'] = False 

    #        if 'seg_areas' not in roidb[img_file]:
    #            roidb[img_file]['seg_areas'] = []
    #        roidb[img_file]['seg_areas'].append((xmax - xmin) * (ymax - ymin))


def make_roidb(img_list_paths):
    roidb = {}
    # 適当なファイルからクラス取得
    df = pd.read_csv(img_list_paths[0], encoding="utf-8", delimiter="\t") 
    class_head = list(df.columns).index("__background__")
    classes = tuple(df.columns)[class_head:]
    # roidbを作成 
    df = pd.concat([pd.read_csv(img_list_path, encoding="utf-8", delimiter="\t") for img_list_path in img_list_paths]) 
    print df.columns
    class_head = list(df.columns).index("__background__")
    classes = tuple(df.columns)[class_head:]
    print classes
    for i, row in enumerate(df.values.tolist()):
        img_path, resize_w, resize_h, x, y, w, h = row[:7]
        # 元の画像サイズ(PILを使えばloadせずに取れるが,そもそもimglistに書いとくべきだったorz)
        im = Image.open(img_path)
        orig_w, orig_h = im.size
        del im
        #bounding boxの座標を変換
        x = int((orig_w / float(resize_w)) * x)
        y = int((orig_h / float(resize_h)) * y)
        w = int((orig_w / float(resize_w)) * w)
        h = int((orig_h / float(resize_h)) * h)
        cls = row[class_head:]
        #if i == 0:
        #    print "remove negative"
        #if cls[0] == 0:
        #    append_roi_2_db(roidb, img_path, cls, x, y, x+w, y+h)
        append_roi_2_db(roidb, img_path, cls, x, y, x+w, y+h)
    #for img_list_path in img_list_paths:
    #    df = pd.read_csv(img_list_path, encoding="utf-8", delimiter="\t") 
    #    class_head = list(df.columns).index("__background__")
    #    classes = tuple(df.columns)[class_head:]
    #    for i, row in enumerate(df.values.tolist()):
    #        img_path, resize_w, resize_h, x, y, w, h = row[:7]
    #        # 元の画像サイズ(PILを使えばloadせずに取れるが,そもそもimglistに書いとくべきだったorz)
    #        im = Image.open(img_path)
    #        orig_w, orig_h = im.size
    #        del im
    #        #bounding boxの座標を変換
    #        x = int((orig_w / float(resize_w)) * x)
    #        y = int((orig_h / float(resize_h)) * y)
    #        w = int((orig_w / float(resize_w)) * w)
    #        h = int((orig_h / float(resize_h)) * h)
    #        cls = row[class_head:]
    #        append_roi_2_db(roidb, img_path, cls, x, y, x+w, y+h)

    for k in roidb.keys():
        roidb[k]["boxes"] = roidb[k]["boxes"].astype(np.uint16)
        #roidb[k]["gt_classes"] = np.array(roidb[k]["gt_classes"], dtype=np.int32)
        overlaps = roidb[k]["gt_overlaps"].astype(np.float32)
        roidb[k]["gt_overlaps"] = scipy.sparse.csr_matrix(overlaps)
        roidb[k]["seg_areas"] = np.array(roidb[k]["seg_areas"], dtype=np.float32)

    return classes, roidb 

