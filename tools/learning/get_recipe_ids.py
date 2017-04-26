# _*_ coding: utf-8 -*-

import gc
import os 
import json
import argparse

import numpy as np
import pandas as pd


RECIPE_IDS_PATH = "/home/fujino/work/output/cookpaddata/recipes_%s.tsv"
OUTPUT_DIR = "/home/fujino/work/output/caffe_db/npy/%s"
STEP_DIR = "/DATA/IMG/step/"
LABEL_DIR = "/home/fujino/work/output/FlowGraph/label/"
RECIPE_IMG_DIR_PATH="/home/fujino/work/output/recipe_image_directory.json"
N_RECIPE = 2500
TEST_DIR = "/home/fujino/work/output/caffe_db/TEST"
SUFFIX = "train" 


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-recipe_ids_path', help=u'recipes_*.tsv',
                        default=RECIPE_IDS_PATH)
    parser.add_argument('-output_dir', help=u'output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('-step_dir', help=u'/DATA/IMG/step/',
                        default=STEP_DIR)
    parser.add_argument('-rcp_loc_step_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default=RECIPE_IMG_DIR_PATH)
    parser.add_argument('-label_dir', help=u'output directory of generate label',
                        default=LABEL_DIR)
    parser.add_argument('-n_recipe', help=u'# of training recipes', type=int, 
                        default=N_RECIPE)
    parser.add_argument('-test_dir', help=u'test recipes directory', 
                        default=TEST_DIR)
    parser.add_argument('-suffix', help=u"suffix of data (When test data is generated, suffix is ''test'')",
                        default = SUFFIX)

    params = parser.parse_args()

    return vars(params)


def check_test_idx(recipe_list, test_dir):
    """
    recipe_listの中で既にテストデータに入っているレシピのindexを返す
    """
    result = np.array([False] * len(recipe_list))
    files = os.listdir(test_dir)
    for input_file in files:
        input_path = os.path.join(test_dir, input_file)
        recipe_df = pd.read_csv(input_path, delimiter='\t', header=None, encoding='utf-8')
        test_recipe_list = recipe_df[0].as_matrix()
        idx = np.in1d(recipe_list, test_recipe_list)
        result = result | idx
        del recipe_df
        gc.collect()

    return result 


def exist_label(recipe_id, label_dir, rcp_loc_step):
    path = os.path.join(label_dir, 
                        rcp_loc_step[recipe_id]["dir"], 
                        recipe_id + "_%s.json" % rcp_loc_step[recipe_id]["steps"][0])
    if os.path.exists(path):
        return True
    else:
        return False


def main(output_dir, recipe_ids_path,
    step_dir,rcp_loc_step_path, label_dir,
    n_recipe, test_dir, suffix):
    print suffix

    print("load...")
    with open(rcp_loc_step_path, 'r') as fin:
        rcp_loc_step = json.load(fin)


    print("load recipes...")
    recipe_df = pd.read_csv(recipe_ids_path, delimiter='\t', header=None, encoding='utf-8')
    recipe_ids = recipe_df[0].values.tolist()

    # ラベル生成済みのみ
    recipe_ids = [r for r in recipe_ids if exist_label(r, label_dir, rcp_loc_step)]
    if len(recipe_ids) < n_recipe:
        print "Recipes lack. %d vs %d" % (len(recipe_ids), n_recipe)
        return 

    # 既にテストデータに入っているものはテストデータにする 
    # 一回テストに入っているものは順次テストにしたほうがマルチラベルの評価がしやすい
    recipe_ids = np.array(recipe_ids) 
    test_idx = check_test_idx(recipe_ids, test_dir)
    if suffix == "test":
        test_recipe_ids = recipe_ids[test_idx]
        print "test", len(test_recipe_ids)
    recipe_ids = recipe_ids[~test_idx]

    if suffix == "test":
        # テストを前の方に
        recipe_ids = np.hstack([test_recipe_ids, recipe_ids])

    if len(recipe_ids) < n_recipe:
        print "Recipes lack. %d vs %d" % (len(recipe_ids), n_recipe)
        return 
    recipe_ids = recipe_ids[:n_recipe]

    print "save..."
    df = pd.DataFrame(recipe_ids)
    df.to_csv(os.path.join(output_dir, 'recipes_%s.tsv' % suffix), 
                sep="\t", encoding="utf-8", index=False, header=None)

    if suffix == "test":
        df.to_csv(os.path.join(test_dir, 'test_%s' % os.path.basename(recipe_ids_path)), 
                                sep="\t", encoding="utf-8", index=False, header=None)


if __name__ == '__main__':
    params = parse()
    main(**params)
