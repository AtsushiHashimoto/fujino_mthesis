# _*_ coding: utf-8 -*-

import numpy as np
import caffe

def load_net(model_path, pretrained_path, mean_path, swap=(0,1,2), scale=1):
    mean_img=np.load(mean_path)
    net = caffe.Classifier(
        model_path,pretrained_path,
        mean=mean_img,
        channel_swap=swap, # RGB=(0,1,2)
        raw_scale = scale) # img in [0,1]
    return net

def load_VGG16_net(model_path, pretrained_path, swap=(0,1,2), scale=1):
    mean_img= np.array([123.68, 116.779, 103.939]) / 255.0
    net = caffe.Classifier(
        model_path,pretrained_path,
        mean=mean_img,
        channel_swap=swap, # RGB=(0,1,2)
        raw_scale = scale) # img in [0,1]
    return net

def extract_feature(net, img, layer="fc7"):
    net.predict([img], oversample=False)
    feature = net.blobs[layer].data.copy()
    feature = feature.flatten()
    return feature
