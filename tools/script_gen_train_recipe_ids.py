# _*_ coding: utf-8 -*-

import os
import get_train_recipe_ids 

ingredients = [
    u"今日の料理",
    u"ニンジン", 
    u"卵", 
    u"ジャガイモ",
    u"豆腐"]
recipe_ids = u"/home/fujino/work/output/cookpaddata/recipes_%s.tsv"
output_dir = u"/home/fujino/work/output/caffe_db/recipe/%s"
old_dir = u"/home/fujino/work/output/caffe_db/npy_old/%s"

#for ingredient in ingredients:
#    input_path = recipe_ids % ingredient 
#    output_ing_dir = output_dir % ingredient
#    old_ing_dir = old_dir % ingredient
#    print input_path
#    if not os.path.exists(output_ing_dir):
#        os.mkdir(output_ing_dir)
#    params = get_train_recipe_ids.parse()
#    params["recipe_ids_path"] = input_path
#    params["output_dir"] = output_ing_dir
#    params["suffix"] = "test" 
#    params["n_recipe"] = 100
#    params["in_img_list_paths"] = [os.path.join(old_ing_dir, "img_list_test.tsv")]
#    params["out_img_list_paths"] = []
#    get_train_recipe_ids.main(**params)
#
#    params["suffix"] = "valid" 
#    params["n_recipe"] = 100
#    params["in_img_list_paths"] = [os.path.join(old_ing_dir, "img_list_valid.tsv")]
#    params["out_img_list_paths"] = [os.path.join(output_ing_dir, "recipes_test.tsv")]
#    get_train_recipe_ids.main(**params)
#
#    params["suffix"] = "train" 
#    params["n_recipe"] = 2500
#    params["in_img_list_paths"] = [os.path.join(old_ing_dir, "img_list_%03d.tsv" % r) for r in range(5)]
#    params["out_img_list_paths"] = [os.path.join(output_ing_dir, "recipes_%s.tsv" % s) for s in ["test", "valid"]]
#    get_train_recipe_ids.main(**params)

ingredients = [
    u"大根", 
    u"ピーマン", 
    u"玉葱",
    u"キャベツ",
    u"トマト",
    u"鳥肉",
    u"ハム",
    u"イカナゴ"
]

for ingredient in ingredients:
    input_path = recipe_ids % ingredient 
    output_ing_dir = output_dir % ("%s_625" % ingredient)
    print input_path
    if not os.path.exists(output_ing_dir):
        os.mkdir(output_ing_dir)
    params = get_train_recipe_ids.parse()
    params["recipe_ids_path"] = input_path
    params["output_dir"] = output_ing_dir
    params["suffix"] = "test" 
    params["n_recipe"] = 100
    params["in_img_list_paths"] = []
    params["out_img_list_paths"] = []
    get_train_recipe_ids.main(**params)

    params["suffix"] = "valid" 
    params["n_recipe"] = 100
    params["in_img_list_paths"] = []
    params["out_img_list_paths"] = [os.path.join(output_ing_dir, "recipes_test.tsv")]
    get_train_recipe_ids.main(**params)

    params["suffix"] = "train" 
    params["n_recipe"] = 625 
    params["in_img_list_paths"] = []
    params["out_img_list_paths"] = [os.path.join(output_ing_dir, "recipes_%s.tsv" % s) for s in ["test", "valid"]]
    get_train_recipe_ids.main(**params)
