# _*_ coding: utf-8 -*-

import os 
import json 
import argparse

import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from module.CookpadRecipe import CookpadRecipe 


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-category_name', help=u'category name', 
                        default=u"今日の料理")
    parser.add_argument('-category_id', help=u'category_id(default:今日の料理)', 
                        default="12b2d6787c17020084c35a7e8734ca7ce3152347")
    parser.add_argument('-recipe_img_dir_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
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


def main(category_name, category_id, recipe_img_dir_path, output_dir, seed,
        usr, password, socket, host, db):
    with open(recipe_img_dir_path, 'r') as fin:
        recipes_image_directory = json.load(fin)

    np.random.seed(seed)

    cookpad_recipe = CookpadRecipe(usr, password, socket, host, db)
    print "recipes..."
    recipes = cookpad_recipe.get_recipe_ids_from_category(category_id)
    recipes = [r for r in recipes if r in recipes_image_directory]
    print "save..." 
    df = pd.DataFrame(recipes)
    df = df.iloc[np.random.permutation(len(df))]
    df.to_csv(os.path.join(output_dir, 'recipes_%s.tsv' % category_name), 
                            sep="\t", encoding="utf-8", index=False, header=None)


if __name__ == '__main__':
    params = parse()
    main(**params)
