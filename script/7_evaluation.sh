CLUSTER=0002
FOODS=ジャガイモ,豆腐
N_FOODS=2

#GROUPING_DIR=grouping
GROUPING_DIR=grouping_master_thesis
#DB_DIR=caffe_db
DB_DIR=caffe_db_test
#RECIPE_DIR=recipes
RECIPE_DIR=recipes_master_thesis

INIT_ITER=4199
FRCNN_ITER=35920
INIT_PREFIX=""
FRCNN_PREFIX=""

python ../tools/predict/predict.py \
	-output_path ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/result.json \
	-model_path ../exp/${DB_DIR}/${CLUSTER}/init/prototxt/deploy.prototxt \
	-pretrained_path ../exp/${DB_DIR}/${CLUSTER}/init/snapshot/${INIT_PREFIX}_iter_${INIT_ITER}.caffemodel \
	-mean_path ../exp/${DB_DIR}/${CLUSTER}/init/train_mean.npy \
	-input_path ../exp/${DB_DIR}/${CLUSTER}/test_annotation/img_list_test_annotation.tsv \
	-step_dir ../external/IMG/step \
	-ss_dir ../exp/selective_search

	-recipe_dir ../exp/${RECIPE_DIR} \
	-use_dirs neg${CLUSTER},${FOODS} \
	-rcp_loc_steps_path ../exp/recipe_location_steps.json \
	-ss_dir ../exp/selective_search

python ../tools/predict/predict_frcnn.py \
	-output_path ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/frcnn/result.json \
	-model_path ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/frcnn/prototxt/test.prototxt \
	-pretrained_path ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/frcnn/snapshot/${FRCNN_PREFIX}_iter_${FRCNN_ITER}.caffemodel \
	-cluster_path ../exp/${GROUPING_DIR}/cluster_${CLUSTER}.txt \
	-input_paths ../exp/${DB_DIR}/${CLUSTER}/test_annotation/img_list_test_annotation.tsv \
	-step_dir ../external/IMG/step 

python ../tools/predict/evaluation_roc.py \
	-output_dir ../exp/${DB_DIR}/${CLUSTER} \
	-n_foods ${N_FOODS} \
	-test_path ../exp/${DB_DIR}/${CLUSTER}/test_annotation/img_list_test_annotation.tsv \
	-result_paths ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/result.json \
	-result_labels VGG-16 \
	-styles k-- \
	-result_paths ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/frcnn/result.json \
	-result_labels FRCNN \
	-styles r- \
	-step_dir ../external/IMG/step 

if [ ! -e ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/result ]; then
	mkdir ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/result
fi
python ../tools/predict/evaluation_save_img.py \
	-output_dir  ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/result \
	-n_foods ${N_FOODS} \
	-test_path  ../exp/${DB_DIR}/${CLUSTER}/test_annotation/img_list_test_annotation.tsv \
	-result_path ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/result.json \
	-step_dir ../external/IMG/step 

if [ ! -e ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/frcnn/result ]; then
	mkdir ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/frcnn/result
fi
python ../tools/predict/evaluation_save_img.py \
	-output_dir  ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/frcnn/result \
	-n_foods ${N_FOODS} \
	-test_path  ../exp/${DB_DIR}/${CLUSTER}/test_annotation/img_list_test_annotation.tsv \
	-result_path ../exp/${DB_DIR}/${CLUSTER}/refine/${INIT_ITER}/frcnn/result.json \
	-step_dir ../external/IMG/step 

