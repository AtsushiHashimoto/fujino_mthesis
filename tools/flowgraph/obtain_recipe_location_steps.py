# _*_ coding: utf-8 -*-

"""
あるIDのレシピのディレクトリと画像(手順番号のリスト)をjson形式で保存
"""

import os 
import argparse
import json


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('step_dir', help=u'COOKPAD_IMAGES/step')
    parser.add_argument('output_dir', help=u'output directory')
    params = parser.parse_args()

    return vars(params)


def main(params):
    step_dir = params["step_dir"]
    output_dir = params["output_dir"]

    base_dirs = os.listdir(step_dir)

    result = {}
    for base_dir in sorted(base_dirs):
        dir = os.path.join(step_dir, base_dir)
        image_files = sorted(os.listdir(dir))
        for i, image_file in enumerate(image_files):
            print "%s %d/%d   \r" % (base_dir, i, len(image_files)), 
            root, _ = os.path.splitext(image_file)
            recipe_id, step_no = root.split('_')
            if recipe_id not in result:
                result[recipe_id] = {} 
                result[recipe_id]['dir'] = base_dir
                result[recipe_id]['steps'] = [step_no]
            else:
                assert base_dir == result[recipe_id]['dir'], "%s is devided into more than two directory! %s %s" % (recipe_id, base_dir, result[recipe_id]['dir'])
                result[recipe_id]['steps'].append(step_no)

    with open(os.path.join(output_dir, "recipe_location_steps.json"), "w") as fout:
        json.dump(result, fout, indent=2, sort_keys=True)


if __name__ == '__main__':
    params = parse()
    main(params)
