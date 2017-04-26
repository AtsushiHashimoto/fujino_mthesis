# _*_ coding: utf-8 -*-

import os 
import sys
import json
import traceback

import cv2
import caffe
import numpy as np
import selectivesearch as ss

"""
AlpacaDB/selectivesearch
Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, right, bottom), #sampleは(x,y,w,h)なんじゃが...
                    'labels': [...]
                },
                ...
            ]
    '''
"""

class ObjectDetection:
    def __init__(self, result_dir="/media/EXT/selective_search"):
        self._result_dir = result_dir

    def detect(self, img_path, overwrite=False, img_size=224, scale=300, sigma=0.7, min_size=10): #パラメータは経験的
        filename = os.path.splitext(os.path.basename(img_path))[0]
        dir_no = os.path.basename(os.path.dirname(img_path))
        save_dir = os.path.join(self._result_dir, dir_no)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, filename) #拡張子なし 

        try:
            img = caffe.io.load_image(img_path, color=True) #caffeのloadは[0,1] 一応動いてるけどもしかしてuint8のほうがいい?
            #img = cv2.imread(img_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if not overwrite and os.path.exists(save_path + ".json"):
                img_lbl = np.load(save_path + ".npy") 
                with open(save_path + ".json", "r") as fin:
                    data = json.load(fin) 
                regions = data["regions"]
                img = caffe.io.resize(img, (data["resize"][1], data["resize"][0], 3)) # resize to fixed size
                #img = cv2.resize(img, (data["resize"][1], data["resize"][0])) # resize to fixed size
                return img, img_lbl, regions
            else:
                img_h, img_w, _ = img.shape
                if img_h >= img_size and img_w >= img_size: #画像が大きい場合はリサイズ(高速化&画像ごとのパラメータ設定をなくす)
                    img = caffe.io.resize(img, (img_size, img_size, 3)) # resize to fixed size
                    #img = cv2.resize(img, (img_size, img_size)) # resize to fixed size
                    img_h, img_w, _ = img.shape
                resize = [img_w, img_h]
                img_lbl, regions = ss.selective_search(img, scale=scale, sigma=sigma, min_size=min_size) #経験的
                np.save(save_path + ".npy", img_lbl)
                data = {"resize":resize, "regions":regions}
                with open(save_path + ".json", "w") as fout:
                    json.dump(data, fout, indent=2)
                return img, img_lbl, regions
        except KeyboardInterrupt:
            print traceback.format_exc(sys.exc_info()[2])
            sys.exit()
        except:
            raise
