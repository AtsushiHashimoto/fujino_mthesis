# _*_ coding: utf-8 -*-


import numpy as np

import caffe
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

class Frcnn:
    """
    private:
        pretraind_path: pretrained.caffemodel
        model_path: model.prototxt
        classes: a tuple of classes (class1, class2, ... )
        nms_thresh: detect more than nms_thresh IoU
        score_thresh: detect more than score_thresh score 
        detection: result of detection
    public:
        net: caffe.Net 
    """
    def __init__(self,
                 pretrained_path,
                 model_path,
                 classes, 
                 gpu = True, gpu_id = 0,
                 pre_nms = 6000, post_nms = 300,
                 nms_thresh = 0.3,
                 score_thresh = 0.01) :
        self._pretrained_path = pretrained_path
        self._model_path = model_path
        self._classes = classes
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        if gpu:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
            cfg.GPU_ID = gpu_id
        else:
            caffe.set_mode_cpu()
        ## Number of top scoring boxes to keep before apply NMS to RPN proposals
        cfg.TEST.RPN_PRE_NMS_TOP_N = pre_nms 
        ## Number of top scoring boxes to keep after applying NMS to RPN proposals
        cfg.TEST.RPN_POST_NMS_TOP_N = post_nms 
        self._nms_thresh = nms_thresh
        self._score_thresh = score_thresh 
        self.net = caffe.Net(self._model_path, self._pretrained_path, caffe.TEST)

    def im_detect(self, img):
        scores, boxes = im_detect(self.net, img) 
        return scores, boxes

    def non_maximum_suppression(self, scores, boxes):
        """
        output: dictionay key: classs; value: a list of [xmin, ymin, xmax, ymax, score] 
        """
        #XMIN_IDX = 0
        #YMIN_IDX = 1
        #XMAX_IDX = 2
        #YMAX_IDX = 3
        SCORE_IDX = 4

        detection = {}
        for cls_idx, cls in enumerate(self._classes[1:]):
            cls_idx += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_idx:4*(cls_idx + 1)]
            cls_scores = scores[:, cls_idx]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self._nms_thresh) # non maximum suppression
            dets = dets[keep, :] # remove overlapped non maximum boxes
            dets = dets[dets[:, SCORE_IDX] > self._score_thresh, :] # remove boxes with low scores
            if dets.size > 0:
                detection[cls] = sorted(dets, key=lambda x : x[SCORE_IDX], reverse=True)
        return detection 

    def detect(self, img):
        """
            register self._detection
        """
        scores, boxes = self.im_detect(img)
        self._detection = self.non_maximum_suppression(scores, boxes)
    
    def get_result(self):
        return self._detection

    def get_top_result(self, cls, top):
        """
            cls: class name
            top: return top N result
        """
        if cls in self._detection:
            if len(self._detection[cls]) >= top:
                result = self._detection[cls][:top]
            else:
                result = self._detection[cls]
            return [list(r) for r in result]

        else:
            return [[0,0,0,0,-1]]
