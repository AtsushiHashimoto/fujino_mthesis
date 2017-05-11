CLUSTER=0002
FOODS_="ジャガイモ 豆腐"
FOODS=ジャガイモ,豆腐

#GROUPING_DIR=grouping
GROUPING_DIR=grouping_master_thesis
RECIPE_DIR=recipes
#RECIPE_DIR=recipes_master_thesis
DB_DIR=caffe_db
#DB_DIR=caffe_db_test

mkdir -p ../exp/${RECIPE_DIR}/TEST
mkdir -p ../exp/selective_search
for food in ${FOODS_}; do
	mkdir -p ../exp/${RECIPE_DIR}/${food}/
done;
mkdir -p ../exp/${RECIPE_DIR}/neg${CLUSTER}
mkdir -p ../exp/caffe_db/${CLUSTER}/init

#学習に用いるレシピIDを決定
echo "学習に用いるレシピIDを決定"
python ../tools/learning/gen_test_train_recipe_ids.py \
	-ingredients ${FOODS} \
	-input_dir ../exp/${RECIPE_DIR} \
	-output_dir ../exp/${RECIPE_DIR} \
	-step_dir ../external/IMG/step \
	-rcp_loc_step_path ../exp/recipe_location_steps.json \
	-label_dir ../exp/flowgraph/label \
	-test_dir ../exp/recipes/TEST \
	-n_train_recipe 625 \
	-n_test_recipe 100 

# 陰性訓練データ用(食材を含まないレシピから選択)
echo "陰性訓練データ用(食材を含まないレシピから選択)"
if [ ! -e ../exp/recipes/neg${CLUSTER} ]; then
	mkdir ../exp/recipes/neg${CLUSTER}
fi
python ../tools/learning/get_recipe_ids_wo_ings.py \
	-recipe_ids_path ../exp/${RECIPE_DIR}/recipes_今日の料理.tsv \
	-output_dir ../exp/${RECIPE_DIR}/neg${CLUSTER} \
	-step_dir ../external/IMG/step \
	-rcp_loc_steps_path ../exp/recipe_location_steps.json \
	-label_dir ../exp/flowgraph/label \
	-n_recipe 100 \
	-test_dir ../exp/recipes/TEST \
	-suffix test \
	-ner_dir ../external/NLP/NER/data \
	-recipe_dir_path ../external/recipe_directory.json \
	-synonym_path ../external/ontology/synonym.tsv \
	-ingredients ${FOODS}

echo "陰性訓練データ用(suffix:train)"
python ../tools/learning/get_recipe_ids_wo_ings.py \
	-recipe_ids_path ../exp/${RECIPE_DIR}/recipes_今日の料理.tsv \
	-output_dir ../exp/recipes/neg${CLUSTER} \
	-step_dir ../external/IMG/step \
	-rcp_loc_steps_path ../exp/recipe_location_steps.json \
	-label_dir ../exp/flowgraph/label \
	-n_recipe 2500 \
	-test_dir ../exp/${RECIPE_DIR}/TEST \
	-suffix train \
	-ner_dir ../external/NLP/NER/data \
	-recipe_dir_path ../external/recipe_directory.json \
	-synonym_path ../external/ontology/synonym.tsv \
	-ingredients ${FOODS}


#修論時のみ 動作確認用 普通にやればいらない
#FOODS=ジャガイモ_625,豆腐_625

# DataBase作成
if [ ! -e ../exp/${DB_DIR}/${CLUSTER} ]; then
	mkdir ../exp/${DB_DIR}/${CLUSTER}
fi

if [ ! -e ../exp/${DB_DIR}/${CLUSTER}/init ]; then
	mkdir ../exp/${DB_DIR}/${CLUSTER}/init
fi
# テスト用のimage listを作成
python ../tools/learning/make_img_list.py \
	-output_dir ../exp/${DB_DIR}/${CLUSTER}/init \
	-rcp_loc_steps_path ../exp/recipe_location_steps.json \
	-recipe_dir ../exp/${RECIPE_DIR} \
	-use_dirs neg${CLUSTER},${FOODS} \
	-cluster_path ../exp/${GROUPING_DIR}/cluster_${CLUSTER}.txt \
	-label_dir ../exp/flowgraph/label \
	-seed 0 \
	-suffix test \
	-step_dir ../external/IMG/step \
	-mode center \
	-ss_dir ../exp/selective_search

# train用のDataBase作成
python ../tools/learning/generate_db.py \
	-output_dir ../exp/${DB_DIR}/${CLUSTER}/init \
	-rcp_loc_steps_path ../exp/recipe_location_steps.json \
	-img_size 224 \
	-data_dir ../exp/${RECIPE_DIR} \
	-use_dirs neg${CLUSTER},${FOODS} \
	-cluster_path ../exp/${GROUPING_DIR}/cluster_${CLUSTER}.txt \
	-label_dir ../exp/flowgraph/label \
	-seed 0 \
	-suffix train \
	-step_dir ../external/IMG/step \
	-mode cover \
	-region 3 \
	-ss_dir ../exp/selective_search
