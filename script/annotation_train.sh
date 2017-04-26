CLUSTER=0002
#GROUPING_DIR=grouping
GROUPING_DIR=grouping_master_thesis

IMG_LIST_SUFFIX=0000

if [ ! -e ../exp/caffe_db/${CLUSTER}/train_annotation ]; then
	mkdir ../exp/caffe_db/${CLUSTER}/train_annotation
fi

python ../tools/annotation/annotation.py \
	-output_dir ../exp/caffe_db/${CLUSTER}/train_annotation
	-cluster_path ../exp/${GROUPING_DIR}/cluster_${CLUSTER}.txt \
	-img_list_path ../exp/caffe_db/${CLUSTER}/init/img_list_train${IMG_LIST_SUFFIX}.tsv \
	-step_dir ../external/IMG/step
	-skip_bg

