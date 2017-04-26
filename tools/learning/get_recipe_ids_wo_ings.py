# _*_ coding: utf-8 -*-

import re
import gc
import os 
import sys
import csv
import json
import argparse

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module import ontology

RECIPE_IDS_PATH = "/home/fujino/work/output/cookpaddata/recipes_%s.tsv" % "今日の料理"
#OUTPUT_DIR = "/home/fujino/work/output/caffe_db/npy/"
OUTPUT_DIR = "/media/EXT/caffe_db/recipe/neg0002"
STEP_DIR = "/DATA/IMG/step/"
LABEL_DIR = "/home/fujino/work/output/FlowGraph/label/"
RECIPE_IMG_DIR_PATH="/home/fujino/work/output/recipe_image_directory.json"
N_RECIPE = 2500
TEST_DIR = "/media/EXT/caffe_db/TEST"
SUFFIX = "train" 

NER_DIR='/DATA/NLP/NER/data/'
RECIPE_DIR_PATH="/home/fujino/work/output/FlowGraph/recipe_directory.json"
SYNONYM_PATH="/home/fujino/work/data/ontology/synonym.tsv"
INGREDIENTS = u"ジャガイモ,豆腐"
IN_RECIPE_PATH = "/media/EXT/caffe_db/recipe/今日の料理/recipes_%s.tsv" % SUFFIX


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-recipe_ids_path', help=u'recipes_*.tsv',
                        default=RECIPE_IDS_PATH)
    parser.add_argument('-output_dir', help=u'output_dir',
                        default=OUTPUT_DIR)
    parser.add_argument('-step_dir', help=u'/DATA/IMG/step/',
                        default=STEP_DIR)
    parser.add_argument('-rcp_loc_steps_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default=RECIPE_IMG_DIR_PATH)
    parser.add_argument('-label_dir', help=u'output directory of generate label',
                        default=LABEL_DIR)
    parser.add_argument('-n_recipe', help=u'# of training recipes', type=int, 
                        default=N_RECIPE)
    parser.add_argument('-test_dir', help=u'test recipes directory', 
                        default=TEST_DIR)
    parser.add_argument('-suffix', help=u"suffix of data (When test data is generated, suffix is ''test'')",
                        default = SUFFIX)

    parser.add_argument('-in_recipe_path', help=u"優先的に含めたいレシピのID",
                        default = IN_RECIPE_PATH)
    parser.add_argument('-ner_dir',
                        default = NER_DIR)
    parser.add_argument('-recipe_dir_path',
                        default = RECIPE_DIR_PATH)
    parser.add_argument('-synonym_path',
                        default = SYNONYM_PATH)
    parser.add_argument('-ingredients',
                        default = INGREDIENTS)

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


def exist_label(recipe_id, label_dir, rcp_loc_steps):
    print(type(recipe_id))
    print(recipe_id)
    print(recipe_id.decode('utf-8'))
    path = os.path.join(label_dir, 
                        rcp_loc_steps[recipe_id]["dir"], 
                        recipe_id + "_%s.json" % rcp_loc_steps[recipe_id]["steps"][0])
    if os.path.exists(path):
        return True
    else:
        return False


def check_recipe_ingredients(recipe_id, ner_dir, recipe_dir, ingredients, synonym):
    def _get_ingredients(file_path):
        """
            input: ingredients_path
            output: a list of ingredients 
        """
        INGREDIENT = 1
    
        r = re.compile(ur'[0-9A-Za-zぁ-んァ-ン一-龥ー]+')
        ingredients = []
        with open(file_path, 'rt') as fin:
            reader = csv.reader(fin, delimiter='\t')
            for row in reader:
                ingredient = row[INGREDIENT]
                ingredient = unicode(ingredient, 'utf-8')
                strs = r.findall(ingredient) # remove symbol
                ingredient = ' '.join(strs)
                ingredient = ingredient.encode('utf-8')
                ingredients.append(ingredient)
    
        return ingredients 


    def _convert_by_synonym(word, partial_match=False):
        if partial_match:
            #正規表現により一番長くマッチするキーの値を返す
            match_word = None
            for synonym_word in synonym.keys():
                match_object = re.search(synonym_word, word)
                if match_object:
                    m_word = match_object.group()
                    if match_word == None or len(m_word) > len(match_word):
                        match_word = m_word
            if match_word != None:
                return synonym[match_word]
            else:
                return None
        elif not partial_match:
            if word in synonym:
                return synonym[word]
            else:
                return None

    ingredients_path = os.path.join(ner_dir, recipe_dir[recipe_id], recipe_id, "ingredients.txt")
    recipe_ingredients =_get_ingredients(ingredients_path)
    recipe_ingredients = [_convert_by_synonym(unicode(i,encoding="utf-8")) for i in recipe_ingredients] 
    recipe_ingredients = set([i for i in recipe_ingredients if not i is None])
    ingredients = set(ingredients)
    intsct = len(ingredients & recipe_ingredients)
    if intsct == 0:
        return True
    else:
        return False 


def main(output_dir, recipe_ids_path, step_dir,rcp_loc_steps_path, label_dir,
    n_recipe, test_dir, suffix,
    in_recipe_path, ner_dir, recipe_dir_path, synonym_path, ingredients):
    print suffix

    print("load...")
    with open(rcp_loc_steps_path, 'r') as fin:
        rcp_loc_steps = json.load(fin)

    with open(recipe_dir_path, 'r') as fin:
        recipe_dir = json.load(fin)

    synonym = ontology.load_food_synonym(synonym_path, key="swing", seasoning=False)

    ingredients = ingredients.split(",")

    print("load recipes...")
    recipe_df = pd.read_csv(recipe_ids_path, delimiter='\t', header=None, encoding='utf-8')
    recipe_ids = recipe_df[0].values.tolist()

    # ラベル生成済みのみ
    recipe_ids = [r for r in recipe_ids if exist_label(r, label_dir, rcp_loc_steps)]
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

    #if len(recipe_ids) < n_recipe:
    #    print "Recipes lack. %d vs %d" % (len(recipe_ids), n_recipe)
    #    return 
    #recipe_ids = recipe_ids[:n_recipe]

    #入れたいものを前に持ってくる
    in_recipe_ids = open(in_recipe_path).readlines()
    in_idx = np.array([r in in_recipe_ids for r in recipe_ids])
    recipe_ids = np.hstack([recipe_ids[in_idx], recipe_ids[~in_idx]])

    #材料が入っていないことを判定
    #n_recipeを超えるかrecipe_idsがなくなるまで繰り返す
    use_recipe_ids = []
    while len(use_recipe_ids) < n_recipe and len(recipe_ids) > 0:
        print len(use_recipe_ids)
        tmp_recipe_ids = recipe_ids[:np.min([n_recipe - len(use_recipe_ids), len(recipe_ids)])]
        recipe_ids = recipe_ids[n_recipe - len(use_recipe_ids):]
        print "check..."
        tmp_recipe_ids = [r for r in tmp_recipe_ids 
                            if check_recipe_ingredients(r, ner_dir, recipe_dir, ingredients, synonym)]
        print "add..."
        use_recipe_ids += tmp_recipe_ids


    print "save..."
    df = pd.DataFrame(use_recipe_ids)
    df.to_csv(os.path.join(output_dir, 'recipes_%s.tsv' % suffix), 
                sep="\t", encoding="utf-8", index=False, header=None)


if __name__ == '__main__':
    params = parse()
    main(**params)
