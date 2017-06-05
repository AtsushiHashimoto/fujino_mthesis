# _*_ coding: utf-8 -*-

import os
import argparse

import get_recipe_ids 

INPUT_DIR = "/home/fujino/work/output/caffe_db/npy"
OUTPUT_DIR = "/home/fujino/work/output/caffe_db/npy"
STEP_DIR = "/DATA/IMG/step/"
LABEL_DIR="/home/fujino/work/output/FlowGraph/label"
N_RECIPE = 1000
TEST_DIR = "/home/fujino/work/output/caffe_db/TEST"


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ingredients', help=u'食材をカンマ区切り',
                        default="")
    parser.add_argument('-input_dir', help=u'recipes_*.tsvが含まれるディレクトリ',
                        default=INPUT_DIR)
    parser.add_argument('-output_dir', help=u'output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('-step_dir', help=u'/DATA/IMG/step/',
                        default=STEP_DIR)
    parser.add_argument('-rcp_loc_step_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default="/home/fujino/work/output/recipe_image_directory.json")
    parser.add_argument('-label_dir', help=u'output directory of generate label',
                        default=LABEL_DIR)
    parser.add_argument('-n_train_recipe', help=u'# of recipes for training', type=int, 
                        default=N_RECIPE)
    parser.add_argument('-n_test_recipe', help=u'# of recipes for test', type=int, 
                        default=N_RECIPE)
    parser.add_argument('-test_dir', help=u'test recipes directory', 
                        default=TEST_DIR)

    params = parser.parse_args()

    return vars(params)


def main(ingredients, input_dir, output_dir, step_dir, rcp_loc_step_path, 
    label_dir, n_train_recipe, n_test_recipe, test_dir):

    ingredients = [unicode(ingredient, encoding="utf-8") for ingredient in ingredients.split(",")]

    for ingredient in ingredients:
        input_path = os.path.join(input_dir, u"recipes_%s.tsv" % ingredient)
        #output_ing_dir = os.path.join(output_dir, u"%s_625" % ingredient)
        output_ing_dir = os.path.join(output_dir, u"%s" % ingredient)

        #print input_path
        if not os.path.exists(output_ing_dir):
            os.mkdir(output_ing_dir)

        #trainにテストを含めないようにtestから
        #testの場合test_dirに保存され, そこのレシピIDは使わない
        #get_recipe_ids.main(input_path, output_ing_dir, step_dir, rcp_loc_step_path,
                            #label_dir, n_test_recipe, test_dir, "test")
        #get_recipe_ids.main(input_path, output_ing_dir, step_dir, rcp_loc_step_path,
                            #label_dir, n_train_recipe, test_dir, "train")
        get_recipe_ids.main(output_ing_dir, input_path, step_dir, rcp_loc_step_path,
                            label_dir, n_test_recipe, test_dir, "test")
        get_recipe_ids.main(output_ing_dir, input_path, step_dir, rcp_loc_step_path,
                            label_dir, n_train_recipe, test_dir, "train")


if __name__ == '__main__':
    params = parse()
    main(**params)
