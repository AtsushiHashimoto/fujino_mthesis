# _*_ coding: utf-8 -*-

import os 
import sys
import argparse

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module.Evaluation import Evaluation
from module.ImgList import ImgList


#CLUSTER = "0015"
#N_FOODS = 1
#TEST_PATH = "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/init"
#RESULT_PATH = "/home/fujino/work/hdf5/0015/init_REG3_P1N4/test_result_7178.json"

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/frcnn_init"
#RESULT_PATH = "/home/fujino/work/hdf5/0015/init_REG3_P1N4/frcnn/test_frcnn_result_35900.json"

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/refine"
#RESULT_PATH = "/home/fujino/work/hdf5/0015/refine_REG3_P1N4/iter7178/test_result_6426.json"

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/frcnn_refine"
#RESULT_PATH = "/home/fujino/work/hdf5/0015/refine_REG3_P1N4/iter7178/frcnn/test_frcnn_result_32140.json"



CLUSTER = "0002"
N_FOODS = 2
TEST_PATH = "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/init"
#RESULT_PATH = "/home/fujino/work/hdf5/0002/init_REG3_P1N4/test_result_29393.json"

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/frcnn_init"
#RESULT_PATH = "/home/fujino/work/hdf5/0002/init_REG3_P1N4/frcnn/test_frcnn_result_42000.json"

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/refine"
#RESULT_PATH = "/home/fujino/work/hdf5/0002/refine_REG3_P1N4/iter29393/test_result_25137.json"

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/frcnn_refine"
#RESULT_PATH = "/home/fujino/work/hdf5/0002/refine_REG3_P1N4/iter29393/frcnn/test_frcnn_result_35920.json"



#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/frcnn_2class"
#RESULT_PATH = "/home/fujino/work/hdf5/potato/refine_REG3_P1N4/iter7254/frcnn/test_frcnn_result_32630.json"

OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0002/frcnn_3class"
RESULT_PATH = "/home/fujino/work/hdf5/0002/refine_REG3_P1N4/iter29393/frcnn/test_frcnn_result_35920.json"


#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/init"
#RESULT_PATH = "/home/fujino/work/hdf5/0015/init_REG3_P1N4/test_result_7178.json"

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/frcnn"
#RESULT_PATH = "/home/fujino/work/hdf5/0015/refine_REG3_P1N4/iter7178/test_result_6426.json"
#
#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/frcnn"
#RESULT_PATH = "/home/fujino/work/hdf5/0015/refine_REG3_P1N4/iter7178/frcnn/test_frcnn_result_32140.json"

#DIR_SUF = "_REG3_P1N4_step"
#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/frcnn_step"
#RESULT_PATH = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4_step/iter7178/frcnn/test_frcnn_result_32140.json" %(CLUSTER)

#TEST_PATH = "/home/fujino/work/hdf5/%s/refine_REG3_P1N4_step/iter7178/frcnn/test_frcnn_result_32140.json" %(CLUSTER)

#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/0015/frcnn2_5k"
#RESULT_PATH = "/media/EXT/caffe_db/hdf5/%s/%s/frcnn%s/test_frcnn2_result_%d.json" % (CLUSTER, REGION, DIR_SUF, ITER)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_dir', help=u'output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('-n_foods', help=u'# of foods', type=int,
                        default=N_FOODS)
    parser.add_argument('-test_path', "-tst", help=u'test img list path',
                        default=TEST_PATH)
    parser.add_argument('-result_path', "-r", help=u'result.json paths',
                        default=RESULT_PATH)
    parser.add_argument('-step_dir', help=u'image directory',
                        default=None)


    params = parser.parse_args()

    return vars(params)


def main(output_dir, test_path, result_path, n_foods, step_dir):
    for food_idx in range(1, n_foods + 1): 
        save_dir = os.path.join(output_dir, "food%04d" % food_idx)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_list = ImgList()
        img_list.load(test_path)
        test_df = img_list.df 
        evaluation = Evaluation(test_df, result_path, food_idx, step_dir=step_dir, n_record=3)

        #evaluation.evaluation_first(food_idx)

        evaluation.save_image(save_dir,  step_dir=step_dir)
        #evaluation.save_image_rank(output_dir,  step_dir=step_dir)
        evaluation.save_image_raw_and_rect(save_dir, food_idx, step_dir=step_dir)
        evaluation.save_image_require(save_dir, food_idx, step_dir=step_dir)

        #tp_rates, fp_rates = evaluation.get_roc_curve()
        #plt.plot(fp_rates, tp_rates)
        #plt.savefig(os.path.join(save_dir, "roc.pdf"))
        #plt.clf()

        #print evaluation.get_ap()
        #print evaluation.get_mean_predict()


if __name__ == '__main__':
    params = parse()
    main(**params)

