CLUSTER=0002

INIT_ITER=4199
PREFIX=""
MAX_ITERS=35920 # 修論時はデータ数=1epoch(仕様によりある画像の複数の領域でバッチを作れるが画像をまたいでバッチを作れない)
SNAPSHOT_ITERS=3591 # 適当 MAX_ITERS / 10とか

python ../tools/learning/train_frcnn.py \
    -gpu_id 0 \
    -name food_recog \
    -db_dir  ../exp/caffe_db/${CLUSTER}/refine/${INIT_ITER} \
    -output_dir ../exp/caffe_db/${CLUSTER}/refine/${INIT_ITER}/frcnn/snapshot \
    -solver_path ../exp/caffe_db/${CLUSTER}/refine/${INIT_ITER}/frcnn/prototxt/solver.prototxt \
    -pretrained_path ../external/VGG16/VGG_ILSVRC_16_layers.caffemodel \
    -max_iters ${MAX_ITERS} \
    -snapshot_iters ${SNAPSHOT_ITERS} \
    -seed 0

