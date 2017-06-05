# _*_ coding: utf-8 -*-

import os
import argparse
import predict_msts
import generate_labels 

NER_DIR = '/DATA/NLP/NER/data/'
COMPG_DIR = '/DATA/NLP/complete-graph/'
FLOW_DIR = "/home/fujino/work/output/FlowGraph/mst/flow"
LABEL_DIR="/home/fujino/work/output/FlowGraph/label"

RECIPE_DIR_PATH = "/home/fujino/work/output/FlowGraph/recipe_directory.json"
SYNONYM_PATH="/home/fujino/work/data/ontology/synonym.tsv"

MAX_RECIPES = 5000
MAX_RECIPES_NEG = 4*MAX_RECIPES
NEGATIVE_RECIPE_CATEGORY = "今日の料理"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ingredients', help=u'ingredient',
                        default="")
    parser.add_argument('-recipe_ids_dir', help=u'directory which contain recipes_(food).tsv',
                        default="/home/fujino/work/output/cookpaddata/")

    parser.add_argument('-ner_dir', help=u'/DATA/NLP/NER/data/', 
                        default=NER_DIR)
    parser.add_argument('-compG_dir', help=u'/DATA/NLP/complete-graph', 
                        default=COMPG_DIR)
    parser.add_argument('-flow_dir', help=u'result of predict msts(flow_dir(/dir_no/flow.csv)', 
                        default=FLOW_DIR)
    parser.add_argument('-label_dir', help=u'result of generate labels(label_dir(/dir_no/flow.csv)',
                        default=LABEL_DIR)
    parser.add_argument('-recipe_dir_path', help=u'FlowGraph/recipe_directory.json',
                        default=RECIPE_DIR_PATH)
    parser.add_argument('-rcp_loc_steps_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default="/home/fujino/work/output/recipe_image_directory.json")
    parser.add_argument('-synonym_path', help=u'ontology/synonym.tsv',
                        default=SYNONYM_PATH)

    parser.add_argument('-max_recipes',  help=u'stop if max recipes are processed', type=int,
                        default=MAX_RECIPES)
    parser.add_argument('-overwrite',  help=u'overwrite', action="store_true",
                        default=False)

    # add by hashimoto
    parser.add_argument('-max_recipes_negative',  help=u'stop if max recipes are processed for negative samples', type=int,
                        default=MAX_RECIPES_NEG)
    parser.add_argument('-negative_recipe_category',help=u'set a recipe category to extract negative samples', default=NEGATIVE_RECIPE_CATEGORY)

    return vars( parser.parse_args() )


def main(ingredients, recipe_ids_dir, ner_dir, compG_dir, flow_dir, label_dir,
        recipe_dir_path, rcp_loc_steps_path, synonym_path, max_recipes, overwrite, max_recipes_negative, negative_recipe_category):

    ingredients = [unicode(ingredient, encoding="utf-8") for ingredient in ingredients.split(",")]

    negative_recipe_category = unicode(negative_recipe_category,'utf-8')
    input_path4negative = os.path.join(recipe_ids_dir, u"recipes_%s.tsv" % negative_recipe_category)

    for ingredient in ingredients:
        input_path = os.path.join(recipe_ids_dir, u"recipes_%s.tsv" % ingredient)
        print ingredient.encode('utf-8')
        _max_recipes = max_recipes
        print(input_path)
        if input_path == input_path4negative:
            _max_recipes = max_recipes_negative
        else:
	    continue
        print "predict msts"
        params = dict(
            recipe_ids_path = input_path,
            ner_dir = ner_dir,
            compG_dir = compG_dir,
            recipe_dir_path = recipe_dir_path,
            output_dir = flow_dir,
            overwrite = overwrite,
            max_recipes = _max_recipes)
        predict_msts.main(**params)

        print "generate labels"
        params = dict(
            recipe_ids_path = input_path,
            ner_dir = ner_dir,
            flow_dir = flow_dir, 
            synonym_path = synonym_path,
            recipe_dir_path = recipe_dir_path,
            rcp_loc_steps_path = rcp_loc_steps_path,
            output_dir = label_dir,
            overwrite = overwrite,
            max_recipes = max_recipes)
        generate_labels.main(**params)


if __name__ == '__main__':
    params = parse()
    main(**params)

