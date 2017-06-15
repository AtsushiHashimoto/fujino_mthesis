FOODS=ニンジン,ジャガイモ,豆腐
MYSQL_USR=fujino
# パスワードを設定した場合は，各コマンドの-passwordオプションを入れる
MYSQL_PASSWORD="fujino"
MYSQL_SOCKET=/var/run/mysqld/mysqld.sock
MYSQL_HOST=localhost
MYSQL_DB=cookpad_data

echo "Msg: Make sure that mysql is alive!"
# 各レシピIDの画像のディレクトリと枚数を保存　
mkdir -p ../exp/recipes
python ../tools/flowgraph/obtain_recipe_location_steps.py ../external/IMG/step ../exp/ 

# 陰性訓練データ用に「今日の料理」カテゴリのレシピIDを取得
python ../tools/flowgraph/obtain_recipes_from_a_category.py \
	-output_dir ../exp/recipes \
	-seed 0 \
	-usr ${MYSQL_USR} \
	-socket ${MYSQL_SOCKET} \
	-host ${MYSQL_HOST} \
	-db ${MYSQL_DB}\
	-password ${MYSQL_PASSWORD} \
	-recipe_img_dir_path ../exp/recipe_location_steps.json \
	-category_name 今日の料理
#	-rcp_loc_steps_path ../exp/recipe_location_steps.json \

# 陽性訓練データ用レシピIDを取得
python ../tools/flowgraph/obtain_recipes_from_ingredients.py \
	-ingredients ${FOODS} \
	-synonym_path ../external/ontology/synonym.tsv \
	-rcp_loc_steps_path ../exp/recipe_location_steps.json \
	-output_dir ../exp/recipes \
	-seed 0 \
	-usr ${MYSQL_USR} \
	-socket ${MYSQL_SOCKET} \
	-host ${MYSQL_HOST} \
	-db ${MYSQL_DB} \
	-password ${MYSQL_PASSWORD} \

max spanning tree(フローグラフ)の推定とフローグラフを辿ってラベルを推定
mkdir -p ../exp/flowgraph/flow/
mkdir -p ../exp/flowgraph/label/
python ../tools/flowgraph/gen_msts_and_labels.py \
	-ingredients ${FOODS},今日の料理 \
	-recipe_ids_dir ../exp/recipes \
	-ner_dir ../external/NLP/NER/data/ \
	-compG_dir ../external/NLP/complete-graph \
	-flow_dir ../exp/flowgraph/flow \
	-label_dir ../exp/flowgraph/label \
	-recipe_dir_path ../external/recipe_directory.json \
	-rcp_loc_steps_path ../exp/recipe_location_steps.json \
	-synonym_path ../external/ontology/synonym.tsv \
	-max_recipes 5000 \
#	-overwrite False 
	-negative_recipe_category "今日の料理" \
	-max_recipes_negative 20000 \
