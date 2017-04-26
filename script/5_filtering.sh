CLUSTER=0002

ITER=4199

#DB_DIR=caffe_db
DB_DIR=caffe_db_test

#スペクトラルクラスタリングによる訓練データ選択
python3 ../tools/learning/python3/exp_clustering_evaluation.py \
	-input_dir ../exp/${DB_DIR}/${CLUSTER}/refine/${ITER} \
	-output_dir ../exp/${DB_DIR}/${CLUSTER}/refine/${ITER}/fp_estimation \
	-label_dir ../exp/flowgraph/label \
	-annotation_dir  ../exp/${DB_DIR}/${CLUSTER}/train_annotation \
	-step_dir ../external/IMG/step \
	-metric linear

python ../tools/learning/update_label_remove.py \
	-output_dir ../exp/${DB_DIR}/${CLUSTER}/refine/${ITER} \
	-f_imgs_path ../exp/${DB_DIR}/${CLUSTER}/refine/${ITER}/f_imgs.npy \
	-db_dir  ../exp/${DB_DIR}/${CLUSTER}/init \
	-fp_est_dir ../exp/${DB_DIR}/${CLUSTER}/refine/${ITER}/fp_estimation

#python ../tools/learning/refine_db_remove.py \
#	-output_dir ../exp/${DB_DIR}/${CLUSTER}/refine/${ITER} \
#	-f_imgs_path ../exp/${DB_DIR}/${CLUSTER}/refine/${ITER}/f_imgs.npy \
#	-db_dir  ../exp/${DB_DIR}/${CLUSTER}/init \
#	-fp_est_dir ../exp/${DB_DIR}/${CLUSTER}/refine/${ITER}/fp_estimation

