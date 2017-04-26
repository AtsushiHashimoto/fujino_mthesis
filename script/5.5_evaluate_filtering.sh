CLUSTER=0002

#DB_DIR=caffe_db
DB_DIR=caffe_db_test

EPOCH=4199
N_EPOCH=5
FIRST_RATE=0.525

if [ ! -e ../exp/${DB_DIR}/${CLUSTER}/fp_estimation ]; then
	mkdir ../exp/${DB_DIR}/${CLUSTER}/fp_estimation
fi

#スペクトラルクラスタリングによる訓練データ選択
python3 ../tools/learning/python3/script_exp_clustering_evaluation.py \
	-input_dir ../exp/${DB_DIR}/${CLUSTER}/refine/ \
	-output_dir ../exp/${DB_DIR}/${CLUSTER}/fp_estimation \
	-label_dir ../exp/flowgraph/label \
	-annotation_dir  ../exp/${DB_DIR}/${CLUSTER}/train_annotation \
	-step_dir ../external/IMG/step \
	-metric linear \
	-epoch ${EPOCH} \
	-n_epoch ${N_EPOCH} 

python ../tools/learning/draw_false_positive_rate.py \
	-output_dir ../exp/${DB_DIR}/${CLUSTER}/fp_estimation \
	-input_path ../exp/${DB_DIR}/${CLUSTER}/fp_estimation/result.csv \
	-first_rate ${FIRST_RATE} \
	-n_row ${N_EPOCH} \
	-ymax 0.7 \
	-food_idx 1
