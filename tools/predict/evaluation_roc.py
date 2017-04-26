# _*_ coding: utf-8 -*-

import os 
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module.Evaluation import Evaluation
from module.ImgList import ImgList

N_FOODS = 2

#CLUSTER = "0015"
#TEST_PATH = "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER
#RESULT_PATHS = [
##"/home/fujino/work/hdf5/%s/frcnn%s/test_frcnn_result_%d.json" % (CLUSTER, DIR_SUF, i) for i in np.arange(5000, 50000+1, 5000)
#"/home/fujino/work/hdf5/%s/refine_REG3_P1N4_step/iter7178/frcnn/test_frcnn_result_32140.json" %(CLUSTER),
#"/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter7178/frcnn/test_frcnn_result_32140.json" %(CLUSTER),
#]
#RESULT_LABELS =[
##"%d"%i for i in np.arange(5000, 45000+1, 5000)
#u"直接ラベル",
#u"遡及ラベル",
#]
#STYLES = [
#"--",
#"-",
#]
#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/%s/" % CLUSTER

#CLUSTER = "0015"
#TEST_PATH = "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER
#RESULT_PATHS = [
#"/home/fujino/work/hdf5/%s/init_REG3_P1N4/test_result_7178.json" %(CLUSTER),
#"/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter7178/test_result_6426.json" %(CLUSTER),
#"/home/fujino/work/hdf5/%s/init_REG3_P1N4/frcnn/test_frcnn_result_35900.json" %(CLUSTER),
#"/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter7178/frcnn/test_frcnn_result_32140.json" %(CLUSTER),
#]
#RESULT_LABELS =[
#u"修正なし+VGG16",
#u"修正あり+VGG16",
#u"修正なし+FRCNN",
#u"修正あり+FRCNN",
#]
#STYLES = [
#"c--",
#"b:",
#"m-.",
#"r-"
#]
#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/%s/" % CLUSTER

CLUSTER = "0002"
TEST_PATH = "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER
RESULT_PATHS = [
"/home/fujino/work/hdf5/%s/init_REG3_P1N4/test_result_29393.json" %(CLUSTER),
"/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter29393/test_result_25137.json" %(CLUSTER),
"/home/fujino/work/hdf5/%s/init_REG3_P1N4/frcnn/test_frcnn_result_42000.json" %(CLUSTER),
"/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter29393/frcnn/test_frcnn_result_35920.json" %(CLUSTER),
]
RESULT_LABELS =[
u"修正なし+VGG16",
u"修正あり+VGG16",
u"修正なし+FRCNN",
u"修正あり+FRCNN",
]
STYLES = [
"c--",
"b:",
"m-.",
"r-"
]
OUTPUT_DIR = "/home/fujino/work/output/food_recognition/%s/" % CLUSTER

#CLUSTER = "0002"
#TEST_PATH = "/home/fujino/work/hdf5/%s/img_list_with_labels_test_annotated.tsv" % CLUSTER
#RESULT_PATHS = [
#"/home/fujino/work/hdf5/potato/refine_REG3_P1N4/iter7254/frcnn/test_frcnn_result_32630.json",
#"/home/fujino/work/hdf5/%s/refine_REG3_P1N4/iter29393/frcnn/test_frcnn_result_35920.json" %(CLUSTER),
#]
#RESULT_LABELS =[
#u"2クラス分類",
#u"3クラス分類",
#]
#STYLES = [
#"--",
#"-",
#]
#OUTPUT_DIR = "/home/fujino/work/output/food_recognition/%s/" % CLUSTER

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_dir', help=u'output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('-n_foods', help=u'# of foods', type=int,
                        default=N_FOODS)
    parser.add_argument('-test_path', "-tst", help=u'test img list path',
                        default=TEST_PATH)
    parser.add_argument('-result_paths', "-rp", help=u'result.json paths', action="append",
                        default=[])
    parser.add_argument('-result_labels', "-rl", help=u'result labels', action="append",
                        default=[])
    parser.add_argument('-styles', help=u'line styles', action="append",
                        default=[])
    parser.add_argument('-step_dir', help=u'image directory',
                        default=None)


    params = parser.parse_args()

    return vars(params)


def main(output_dir, test_path, result_paths, result_labels, styles, n_foods, step_dir):
    for food_idx in range(1, n_foods+1):
        print food_idx
        fp = FontProperties(fname=r'/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf', size=14)
        img_list = ImgList()
        img_list.load(test_path)
        test_df = img_list.df 
        plt.figure(figsize=(6,4))
        aps = []
        mean_predicts = []
        require = []
        for r_path, r_label, style in zip(result_paths, result_labels, styles):
            print r_label
            evaluation = Evaluation(test_df, r_path, food_idx, step_dir=step_dir)
            tp_rates, fp_rates = evaluation.get_roc_curve()
            ap = evaluation.get_ap()
            mp = evaluation.get_mean_predict()
            plt.plot(fp_rates, tp_rates, style, label=r_label, lw=2)
            aps.append(ap)
            mean_predicts.append(mp)
            tp_rates = np.sort(np.array(tp_rates))
            fp_rates = np.sort(np.array(fp_rates))
            req_ids = (fp_rates >= 0.2)
            req = np.mean([tp_rates[np.where(req_ids)[0][0]], tp_rates[np.where(~req_ids)[0][-1]]]) # 0.2の境目の平均
            require.append(req)

        plt.vlines(x = 0.2, ymin=0, ymax = 1, colors="k", linestyles="--")
        plt.legend(loc="lower right", prop=fp)
        plt.savefig(os.path.join(output_dir, "roc_%03d.pdf"%food_idx))
        plt.clf()

        print "average precision"
        for l, ap in zip(result_labels, aps):
            print "%s\t%.3f" % (l, ap)
        #print "mean likelihoods"
        #for l, mp in zip(result_labels, mean_predicts):
        #    print "%s\t%.3f" % (l, mp)
        print "require"
        for l, r in zip(result_labels, require):
            print "%s\t%.3f" % (l, r)


if __name__ == '__main__':
    params = parse()
    main(**params)

