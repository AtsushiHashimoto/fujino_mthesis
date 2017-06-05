CLUSTER=0002

ITER=4199
PREFIX=""

#学習
python ../tools/learning/train.py \
	-solver_path ../exp/caffe_db/${CLUSTER}/init/prototxt/solver.prototxt \
	-pretrained_path ../external/VGG16/VGG_ILSVRC_16_layers.caffemodel \
	#-solverstate_path #途中から動かす時

#特徴抽出
if [ ! -e ../exp/caffe_db/${CLUSTER}/refine ]; then
	mkdir ../exp/caffe_db/${CLUSTER}/refine
fi

#if [ ! -e ../exp/caffe_db/${CLUSTER}/refine/${ITER} ]; then
#	mkdir ../exp/caffe_db/${CLUSTER}/refine/${ITER}
#fi

python ../tools/learning/extract_features.py \
	-output_dir ../exp/caffe_db/${CLUSTER}/refine/${ITER} \
	-model_path ../exp/caffe_db/${CLUSTER}/init/prototxt/deploy.prototxt \
	-pretrained_path ../exp/caffe_db/${CLUSTER}/init/snapshot/${PREFIX}_iter_${ITER}.caffemodel \
	-mean_path ../exp/caffe_db/${CLUSTER}/init/train_mean.npy \
	-db_dir  ../exp/caffe_db/${CLUSTER}/init 

