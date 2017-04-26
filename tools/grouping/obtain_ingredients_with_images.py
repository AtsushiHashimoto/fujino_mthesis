# _*_ coding: utf-8 -*-

import os 
import json 
import argparse
import pandas as pd

import MySQLdb 
from CookpadRecipe import CookpadRecipe

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-synonym_path', help=u'ontology/synonym.tsv',
                        default="/home/fujino/work/data/ontology/synonym.tsv")
    parser.add_argument('-recipes_image_directory_path', help=u'recipe_id:dictionary("dir":directory, "steps":list of step_no)',
                        default="/home/fujino/work/output/cookpaddata/recipe_image_directory.json")
    parser.add_argument('output_dir', help=u'output directory')
    params = parser.parse_args()

    return vars(params)


def read_synonym_reverse(synonym_path):
    synonyms = pd.read_csv(synonym_path, delimiter='\t', header = None, encoding='utf-8')
    idx = [l not in [u'動作', u'調理器具'] for l in list(synonyms.iloc[:, 0])]
    assert True in idx and False in idx
    ontology = {} 
    for ingredient, synonym in zip(synonyms.iloc[idx, 1], synonyms.iloc[idx, 2]):
        if ingredient not in ontology:
            ontology[ingredient] = []
        ontology[ingredient].append(synonym)
    return ontology


def get_ingredients(recipe_id, recipe_dir, ner_dir, ontology, enum):
    if enum[0] % 50 == 0:
        print "%d / %d" % enum 
    dir_no = recipe_dir[recipe_id]
    ingredients_path = os.path.join(ner_dir, dir_no, recipe_id, 'ingredients.txt')
    ingredients = get_ingredients_from_file(ingredients_path)
    ingredients = cleaning_ingredients(ingredients, ontology)
    return ingredients


def main(params):
    recipes_image_directory_path = params['recipes_image_directory_path']
    synonym_path = params["synonym_path"]
    output_dir = params['output_dir']

    print "load..."
    ontology = read_synonym_reverse(synonym_path)
    with open(recipes_image_directory_path, 'r') as fin:
        recipes_image_directory = json.load(fin)

    connection = MySQLdb.connect(host='localhost', 
                                db='cookpad_data', 
                                user='fujino', 
                                passwd='', 
                                charset='utf8', 
                                unix_socket='/var/run/mysqld/mysqld.sock')
    cookpad_recipe = CookpadRecipe(connection)

    counter = {}
    for ingredient, ingredients in ontology.items():
        print ingredient
        print " ".join(ingredients)
        recipes = set(cookpad_recipe.get_recipe_ids_from_ingredients(ingredients))
        recipes = recipes & set(recipes_image_directory.keys()) 
        print len(recipes) 
        counter[ingredient] = len(recipes)

    print "save..."
    result = [list(c) for c in sorted(counter.items(), key = lambda x : x[1], reverse=True)]
    df = pd.DataFrame(result)
    df.columns = [u"ingredient", u"no of recipes"]
    df.to_csv(os.path.join(output_dir, 'ingredients_with_imgs.csv'), encoding="utf-8", index=False)


if __name__ == '__main__':
    params = parse()
    main(params)
