# _*_ coding: utf-8 -*-

import os 
import sys
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module.Evaluation import Evaluation
from module.ImgList import ImgList


#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/frcnn_2vs3"
#RESULT_PATH1 = "/home/fujino/work/hdf5/potato/refine_REG3_P1N4/iter7254/frcnn/test_frcnn_result_32630.json"
#RESULT_PATH2 = "/home/fujino/work/hdf5/0002/refine_REG3_P1N4/iter29393/frcnn/test_frcnn_result_35920.json"

CLUSTER = "0015"
TEST_PATH = "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER 
N_FOODS = 1

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/vs_init_refine"
#RESULT_PATH1 = "/home/fujino/work/hdf5/%s/init_REG3_P1N4/test_result_7178.json" %(CLUSTER)
#RESULT_PATH2 = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter7178/test_result_6426.json" %(CLUSTER)

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/vs_init_frcnn"
#RESULT_PATH1 = "/home/fujino/work/hdf5/%s/init_REG3_P1N4/test_result_7178.json" %(CLUSTER)
#RESULT_PATH2 = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter7178/frcnn/test_frcnn_result_32140.json" %(CLUSTER)

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/vs_text_flow"
#RESULT_PATH1 = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4_step/iter7178/frcnn/test_frcnn_result_32140.json" %(CLUSTER)
#RESULT_PATH2 = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter7178/frcnn/test_frcnn_result_32140.json" %(CLUSTER)

OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/vs_fi_fr"
RESULT_PATH1 = "/home/fujino/work/hdf5/0015/init_REG3_P1N4/frcnn/test_frcnn_result_35900.json"
RESULT_PATH2 = "/home/fujino/work/hdf5/0015/refine_REG3_P1N4/iter7178/frcnn/test_frcnn_result_32140.json"


#CLUSTER = "0002"
#TEST_PATH = "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER
#N_FOODS = 2

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/vs_init_refine"
#RESULT_PATH1 = "/home/fujino/work/hdf5/%s/init_REG3_P1N4/test_result_29393.json" %(CLUSTER)
#RESULT_PATH2 = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter29393/test_result_25137.json" %(CLUSTER)

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/vs_init_frcnn"
#RESULT_PATH1 = "/home/fujino/work/hdf5/%s/init_REG3_P1N4/test_result_29393.json" %(CLUSTER)
#RESULT_PATH2 = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter29393/frcnn/test_frcnn_result_35920.json" %(CLUSTER)

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/frcnn_2vs3"
#RESULT_PATH1 = "/home/fujino/work/hdf5/potato/refine_REG3_P1N4/iter7254/frcnn/test_frcnn_result_32630.json"
#RESULT_PATH2 = "/home/fujino/work/hdf5/0002/refine_REG3_P1N4/iter29393/frcnn/test_frcnn_result_35920.json"


#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/vs_fi_fr"
#RESULT_PATH1 = "/home/fujino/work/hdf5/%s/init_REG3_P1N4/frcnn/test_frcnn_result_42000.json" %(CLUSTER)
#RESULT_PATH2 = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter29393/frcnn/test_frcnn_result_35920.json" %(CLUSTER)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_dir', help=u'output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('-n_foods', help=u'# of foods',
                        default=N_FOODS)
    parser.add_argument('-test_path', "-tst", help=u'test img list path',
                        default=TEST_PATH)
    parser.add_argument('-result_path1',  help=u'result.json paths',
                        default=RESULT_PATH1)
    parser.add_argument('-result_path2',  help=u'result.json paths',
                        default=RESULT_PATH2)
    parser.add_argument('-step_dir', help=u'image directory',
                        default=None)


    params = parser.parse_args()

    return vars(params)


def main(output_dir, test_path, result_path1, result_path2, n_foods, step_dir):
    os.mkdir(output_dir)
    for food_idx in range(1, n_foods + 1): 
        save_dir = os.path.join(output_dir, "food%04d" % food_idx)
        os.mkdir(save_dir)
        img_list = ImgList()
        img_list.load(test_path)
        test_df = img_list.df 

        evaluation1 = Evaluation(test_df, result_path1, food_idx, step_dir=step_dir, n_record=3)
        evaluation2 = Evaluation(test_df, result_path2, food_idx, step_dir=step_dir, n_record=3)

        #tp_rates1, fp_rates1 = evaluation1.get_roc_curve()
        #tp_rates2, fp_rates2 = evaluation2.get_roc_curve()
        #tp_rates1 = np.sort(np.array(tp_rates1))
        #fp_rates1 = np.sort(np.array(fp_rates1))
        #tp_rates2 = np.sort(np.array(tp_rates2))
        #fp_rates2 = np.sort(np.array(fp_rates2))
        #max_diff = 0
        #max_fp = 0
        #for r in np.arange(0.01,1,0.01):
        #    req_ids = (fp_rates1 >= r)
        #    if len(np.where(req_ids)[0]) > 0 and len(np.where(~req_ids)[0]):
        #        req1 = np.mean([tp_rates1[np.where(req_ids)[0][0]], tp_rates1[np.where(~req_ids)[0][-1]]])
        #        req_ids = (fp_rates2 >= r)
        #        if len(np.where(req_ids)[0]) > 0 and len(np.where(~req_ids)[0]):
        #            req2 = np.mean([tp_rates2[np.where(req_ids)[0][0]], tp_rates2[np.where(~req_ids)[0][-1]]])
        #            if np.abs(req1 - req2) > max_diff:
        #                max_diff = np.abs(req1 - req2)
        #                max_fp = r
        #print max_fp
        #idxs1 = evaluation1.get_image_req_idx(food_idx, req=max_fp)
        #idxs2 = evaluation2.get_image_req_idx(food_idx, req=max_fp)

        idxs1 = evaluation1.get_image_req_idx(food_idx)
        idxs2 = evaluation2.get_image_req_idx(food_idx)
        print [len(j) for i in idxs1 for j in i]
        print [len(j) for i in idxs2 for j in i]

        ssave_dir = os.path.join(save_dir, "O1pos2neg")
        os.mkdir(ssave_dir)
        idxs = list(set(idxs1[0][0]) - set(idxs2[0][0]))
        ss_dir = os.path.join(ssave_dir, "1")
        os.mkdir(ss_dir)
        evaluation1.save_image_idxs(ss_dir, food_idx, idxs, colors=[(0,255,0)])
        ss_dir = os.path.join(ssave_dir, "2")
        os.mkdir(ss_dir)
        evaluation2.save_image_idxs(ss_dir, food_idx, idxs, colors=[(0,0,255)])

        ssave_dir = os.path.join(save_dir, "O1neg2pos")
        os.mkdir(ssave_dir)
        idxs = list(set(idxs2[0][0]) - set(idxs1[0][0]))
        ss_dir = os.path.join(ssave_dir, "1")
        os.mkdir(ss_dir)
        evaluation1.save_image_idxs(ss_dir, food_idx, idxs, colors=[(0,0,255)])
        ss_dir = os.path.join(ssave_dir, "2")
        os.mkdir(ss_dir)
        evaluation2.save_image_idxs(ss_dir, food_idx, idxs, colors=[(0,255,0)])

        ssave_dir = os.path.join(save_dir, "X1pos2neg")
        os.mkdir(ssave_dir)
        idxs = list(set(idxs1[1][0]) - set(idxs2[1][0]))
        ss_dir = os.path.join(ssave_dir, "1")
        os.mkdir(ss_dir)
        evaluation1.save_image_idxs(ss_dir, food_idx, idxs, colors=[(0,255,0)])
        ss_dir = os.path.join(ssave_dir, "2")
        os.mkdir(ss_dir)
        evaluation2.save_image_idxs(ss_dir, food_idx, idxs, colors=[(0,0,255)])

        ssave_dir = os.path.join(save_dir, "X1neg2pos")
        os.mkdir(ssave_dir)
        idxs = list(set(idxs2[1][0]) - set(idxs1[1][0]))
        ss_dir = os.path.join(ssave_dir, "1")
        os.mkdir(ss_dir)
        evaluation1.save_image_idxs(ss_dir, food_idx, idxs, colors=[(0,0,255)])
        ss_dir = os.path.join(ssave_dir, "2")
        os.mkdir(ss_dir)
        evaluation2.save_image_idxs(ss_dir, food_idx, idxs, colors=[(0,255,0)])



if __name__ == '__main__':
    params = parse()
    main(**params)

