# _*_ coding: utf-8 -*-

import io
import os 
import sys
import glob
import json
import argparse
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module import ontology
from mst import generate_label


NER_DIR='/DATA/NLP/NER/data/'
FLOW_DIR='/home/fujino/work/output/FlowGraph/mst/flow/'
SYNONYM_PATH="/home/fujino/work/data/ontology/synonym.tsv"
RECIPE_DIR_PATH="/home/fujino/work/output/FlowGraph/recipe_directory.json"
OUTPUT_DIR="/home/fujino/work/output/FlowGraph/label"
OVERWRITE=False
MAX_RECIPES=10000


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('recipe_ids_path', help=u'recipe_ids.txt')
    parser.add_argument('-ner_dir', help=u'/DATA/NLP/NER/data/', 
                        default=NER_DIR)
    parser.add_argument('-flow_dir', help=u'result of predict msts(flow_dir(/dir_no/flow.csv)', 
                        default=FLOW_DIR)
    parser.add_argument('-synonym_path', help=u'ontology/synonym.tsv',
                        default=SYNONYM_PATH)
    parser.add_argument('-recipe_dir_path', help=u'FlowGraph/recipe_dir.json',
                        default=RECIPE_DIR_PATH)
    parser.add_argument('-rcp_loc_steps_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default="/home/fujino/work/output/recipe_image_directory.json")
    parser.add_argument('-output_dir', help=u'output directory',
                        default=OUTPUT_DIR)
    parser.add_argument('-overwrite', help=u'overwrite', action="store_true",
                        default=OVERWRITE)
    parser.add_argument('-max_recipes',  help=u'stop if max recipes are processed', type=int,
                        default=MAX_RECIPES)

    params = parser.parse_args()

    return vars(params)


def main( ner_dir, flow_dir, synonym_path,
    recipe_ids_path, recipe_dir_path, rcp_loc_steps_path, output_dir, overwrite, max_recipes):

    synonym = ontology.load_food_synonym(synonym_path)

    with open(recipe_dir_path, 'r') as fin:
        recipe_dir = json.load(fin)
    with open(rcp_loc_steps_path, 'r') as fin:
        rcp_loc_steps = json.load(fin)

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
        flow_path = os.path.join(flow_dir, dir_no, '%s.csv' % recipe_id)
        ingredients_path = os.path.join(ner_dir, dir_no, recipe_id, 'ingredients.txt')
        if os.path.exists(flow_path):
            image_dir = rcp_loc_steps[recipe_id]['dir']
            output_image_dir = os.path.join(output_dir, image_dir)
            if not overwrite: # ファイルが一つでも保存されていれば処理済みとする
                save_files = os.path.join(output_image_dir, '%s_*.json'%recipe_id)
                save_files = glob.glob(save_files)
            if overwrite or len(save_files) == 0:
                try:
                    flow_df = generate_label.get_dataframe_of_flowgraph(flow_path)
                    ingredients = generate_label.get_ingredients(ingredients_path)
                    label = generate_label.generate_label(flow_df, ingredients, synonym)
                    # 出力ディレクトリは画像フォルダと対応
                    if not os.path.exists(output_image_dir):
                        os.mkdir(output_image_dir)
                    for step in label.keys():
                        with io.open(os.path.join(output_image_dir, '%s_%s.json'%(recipe_id,step)), 'w', encoding='utf-8') as fout:
                            data = json.dumps(label[step], fout, indent=2, sort_keys=True, ensure_ascii=False)
                            fout.write(unicode(data)) # auto-decodes data to unicode if str
                    finished += 1
                except KeyboardInterrupt:
                    print traceback.format_exc(sys.exc_info()[2])
                    sys.exit()
                except:
                    print recipe_id 
                    print traceback.format_exc(sys.exc_info()[2])
            elif len(save_files) > 0:
                finished += 1
        if finished >= max_recipes:
            break


if __name__ == '__main__':
    params = parse()
    main(**params)
    
