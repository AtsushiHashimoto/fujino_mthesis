# _*_ coding: utf-8 -*-
# Python 3.x

import os 
import argparse
import exp_clustering_evaluation as exp
import pandas as pd 

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', help=u'input directory which contains features.npy',
                        default = None)
    parser.add_argument('-output_dir', help=u'output directory',
                        default = None)
    parser.add_argument('-label_dir', help=u'label directory',
                        default = None)
    parser.add_argument('-annotation_dir', help=u'img_list_train*_annotation.tsvが入ったディレクトリ',
                        default = "")
    parser.add_argument('-step_dir', help=u'image directory',
                        default = None)
    parser.add_argument('-metric', help=u'metric of similarities',
                        default = "linear")
    parser.add_argument('-epoch', help=u'1エポックのイテレーション回数', type=int,
                        default = 0)
    parser.add_argument('-n_epoch', help=u'何エポックまで評価するか', type=int,
                        default = 0)

    params = parser.parse_args()

    return vars(params)


def main(input_dir, output_dir, label_dir, annotation_dir, step_dir, metric, epoch, n_epoch):
    params = {}
    params["label_dir"] = label_dir 
    params["annotation_dir"] = annotation_dir 
    params["step_dir"] = step_dir 
    params["metric"] = metric
    result = []
    for i in range(1,n_epoch+1):
        #params["input_dir"] = os.path.join(input_dir, "iter%d"%(epoch*i))
        params["input_dir"] = os.path.join(input_dir, "%d"%(epoch*i))
        params["output_dir"] = os.path.join(output_dir, "fp_estimation_%d_%s" % (epoch*i, metric))
    
        r = exp.main(**params)
        result.append([i] + r)
    result = pd.DataFrame(result)
    result.to_csv(os.path.join(output_dir, "result.csv"), index=False, header=False)


if __name__ == "__main__":
    params = parse()
    main(**params)

