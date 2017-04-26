# _*_ coding: utf-8 -*-

import os
import sys
import json
import argparse
import traceback

from mst import predict_mst

NER_DIR = '/DATA/NLP/NER/data/'
COMPG_DIR = '/DATA/NLP/complete-graph/'
RECIPE_DIRECTORY_PATH = "/home/fujino/work/output/FlowGraph/recipe_directory.json"
OUTPUT_DIR = "/home/fujino/work/output/FlowGraph/mst/flow"
OVERWRITE = False
MAX_RECIPES = 10000


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('recipe_ids_path', help=u'recipe_ids.tsv')
    parser.add_argument('-ner_dir', help=u'/DATA/NLP/NER/data/', 
                        default=NER_DIR)
    parser.add_argument('-compG_dir', help=u'/DATA/NLP/complete-graph', 
                        default=COMPG_DIR)
    parser.add_argument('-recipe_dir_path', help=u'FlowGraph/recipe_directory.json',
                        default=RECIPE_DIRECTORY_PATH)
    parser.add_argument('-output_dir', help=u'output directory',
                        default=OUTPUT_DIR)
    parser.add_argument('-overwrite',  help=u'overwrite', action="store_true",
                        default=OVERWRITE)
    parser.add_argument('-max_recipes',  help=u'stop if max recipes are processed', type=int,
                        default=MAX_RECIPES)

    params = parser.parse_args()

    return vars(params)


def main(ner_dir, compG_dir, recipe_ids_path, recipe_dir_path,
    output_dir, overwrite, max_recipes):

    with open(recipe_dir_path, 'r') as fin:
        recipe_dir = json.load(fin)

    with open(recipe_ids_path, 'r') as fin:
        recipe_ids = fin.readlines()
    recipe_ids = [recipe_id.strip() for recipe_id in recipe_ids]
    recipe_ids = [recipe_id for recipe_id in recipe_ids if recipe_id in recipe_dir] # exist file in NER directory

    finished = 0

    for i, recipe_id in enumerate(recipe_ids):
        if i % 100 == 0:
            print '\r%d/%d  %d/%d   '%(finished, max_recipes, i, len(recipe_ids)),
            sys.stdout.flush()
        dir_no = recipe_dir[recipe_id]
        compG_path = os.path.join(compG_dir, dir_no, recipe_id + '.mst-parse') 
        if os.path.exists(compG_path): # exist file in complete-graph directory
            ner_path = os.path.join(ner_dir, dir_no, recipe_id, 'step_final_memos.txt')
            output_no_dir = os.path.join(output_dir, dir_no)
            if not os.path.exists(output_no_dir):
                os.mkdir(output_no_dir)
            try:
                predict_mst.generate_mst(ner_path, compG_path, output_no_dir, overwrite)
                finished += 1
            except KeyboardInterrupt:
                print traceback.format_exc(sys.exc_info()[2])
                sys.exit()
            except:
                print recipe_id 
                print traceback.format_exc(sys.exc_info()[2])
        if finished >= max_recipes:
            break


if __name__ == '__main__':
    params = parse()
    main(**params)
