CLUSTER=0002
#GROUPING_DIR=grouping
GROUPING_DIR=grouping_master_thesis

if [ ! -e ../exp/caffe_db/${CLUSTER}/test_annotation ]; then
	mkdir ../exp/caffe_db/${CLUSTER}/test_annotation
fi

python ../tools/annotation/annotation.py \
	-output_dir ../exp/caffe_db/${CLUSTER}/test_annotation
	-cluster_path ../exp/${GROUPING_DIR}/cluster_${CLUSTER}.txt \
	-img_list_path ../exp/caffe_db/${CLUSTER}/init/img_list_test.tsv \
	-step_dir ../external/IMG/step

