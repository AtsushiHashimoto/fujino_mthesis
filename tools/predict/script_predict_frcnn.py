# _*_ coding: utf-8 -*-

import predict_frcnn as pf 
import numpy as np

DIR_SUF = "_REG3_P1N4"

clusters = ["0002"]
snapshot_prefix=["frcnn_"]
#clusters = ["potato"]
#snapshot_prefix=["frcnn_10k"]
iters = np.arange(5000, 45000+1, 5000)

params = {} 
params["step_dir"] = None
for cls, pre in zip(clusters, snapshot_prefix):
    params["cluster_path"] = "/home/fujino/work/output/clustering/cluster_%s.txt" % cls 
    params["input_paths"] = [ "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % cls ]
    #params["input_paths"] = [ "/home/fujino/work/hdf5/0002/img_list_with_labels_test_annotated.tsv"]
    for ite in iters:
        #params["output_path"] = "/home/fujino/work/hdf5/%s/frcnn%s/test_frcnn_result_%d.json" % (cls, DIR_SUF, ite)
        #params["model_path"] = "/home/fujino/work/hdf5/%s/frcnn%s/prototxt/test.prototxt" % (cls, DIR_SUF)
        #params["pretrained_path"] = "/home/fujino/work/hdf5/%s/frcnn%s/snapshot/%s_iter_%d.caffemodel" % (cls, DIR_SUF, pre, ite)

        params["output_path"] = "/home/fujino/work/hdf5/%s/refine_remove%s/iter15000/frcnn/test_frcnn_result_%d.json" % (cls, DIR_SUF, ite)
        params["model_path"] = "/home/fujino/work/hdf5/%s/refine_remove%s/iter15000/frcnn/prototxt/test.prototxt" % (cls, DIR_SUF)
        params["pretrained_path"] = "/home/fujino/work/hdf5/%s/refine_remove%s/iter15000/frcnn/snapshot/%s_iter_%d.caffemodel" % (cls, DIR_SUF, pre, ite)

        pf.main(**params)

