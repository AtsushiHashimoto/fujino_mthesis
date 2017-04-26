# _*_ coding: utf-8 -*-

import os 
import json 
import argparse

import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module import ontology
from module.CookpadRecipe import CookpadRecipe 


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ingredients', help=u'ingredient',
                        default="")
    parser.add_argument('-synonym_path', help=u'ontology/synonym.tsv',
                        default="/home/fujino/work/data/ontology/synonym.tsv")
    parser.add_argument('-rcp_loc_steps_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default="/home/fujino/work/output/recipe_image_directory.json")
    parser.add_argument('-output_dir', help=u'output directory',
                        default="/home/fujino/work/output/cookpaddata")
    parser.add_argument('-seed', help=u'numpy random seed', type=int,
                        default=0)

    parser.add_argument('-usr', help=u'mysql usr',
                        default="fujino")
    parser.add_argument('-password', help=u'mysql password',
                        default="")
    parser.add_argument('-socket', help=u'mysql socket',
                        default="/var/run/mysqld/mysqld.sock")
    parser.add_argument('-host', help=u'mysql host',
                        default="localhost")
    parser.add_argument('-db', help=u'mysql db',
                        default="cookpad_data")
    params = parser.parse_args()

    return vars(params)


def main(ingredients, synonym_path, rcp_loc_steps_path, output_dir, seed,
        usr, password, socket, host, db):
    synonym = ontology.load_food_synonym(synonym_path, key="concept")
    with open(rcp_loc_steps_path, 'r') as fin:
        rcp_loc_steps = json.load(fin)

    cookpad_recipe = CookpadRecipe(usr, password, socket, host, db)

    ingredients = [unicode(ingredient, encoding="utf-8") for ingredient in ingredients.split(",")]

    np.random.seed(seed)

    def prn(x):
        print x.encode('utf-8')
    for ingredient in ingredients:
        print ingredient.encode('utf-8')
	#map(prn,synonym.keys())
        swings = synonym[ingredient]
        print " ".join(swings).encode('utf-8')
        recipes = cookpad_recipe.get_recipe_ids_from_ingredients(swings)
        # save recipes which contain more than one images
        recipes = [r for r in recipes if r in rcp_loc_steps]
        if len(recipes) > 0:
            print "save..." 
            df = pd.DataFrame(recipes)
            df = df.iloc[np.random.permutation(len(df))]
            df.to_csv(os.path.join(output_dir, 'recipes_%s.tsv' % ingredient), 
                                    sep="\t", encoding="utf-8", index=False, header=None)
        else:
            print "There are no recipes which contains", swings


if __name__ == '__main__':
    params = parse()
    main(**params)
