# _*_ coding: utf-8 -*-

import argparse

import caffe

CLUSTER = "0015"
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-solver_path", "-s", help="/path/to/solver.prototxt",
                        default="/home/fujino/work/hdf5/%s/init_REG5_P1N4/prototxt/solver.prototxt" % (CLUSTER))
    parser.add_argument("-pretrained_path", "-p", help="/path/to/pretrained.caffemodel",
                        default='/home/fujino/work/data/caffe/VGG_ILSVRC_16_layers.caffemodel')
    parser.add_argument("-solverstate_path", "-ss", 
                        default=None)
                        #default="/home/fujino/work/hdf5/%s/init_REG5_P1N4/snapshot/_iter_72814.solverstate" % (CLUSTER))
                        #default="/media/EXT/caffe_db/hdf5/%s/%s/init_P1N4_BG300_REG3/snapshot/_iter_8000.solverstate" % (CLUSTER, REGION))
                        #default="/media/EXT/caffe_db/hdf5/%s/%s/init/snapshot/_iter_100000.solverstate" %(CLUSTER, REGION))
    params = parser.parse_args()

    return vars(params)


def main(solver_path, pretrained_path, solverstate_path):

    caffe.set_mode_gpu()
    caffe.set_device(0) #gpu_id
    solver = caffe.get_solver(solver_path)
    if solverstate_path: #途中から
        solver.restore(solverstate_path)
    else:
        solver.net.copy_from(pretrained_path)
    solver.solve()


if __name__ == '__main__':
    params = parse()
    main(**params)
