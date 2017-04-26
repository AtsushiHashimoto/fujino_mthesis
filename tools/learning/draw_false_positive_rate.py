# _*_ coding: utf-8 -*-

import os 
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_dir', help=u'output_dir', default=None)
    parser.add_argument('-input_path', help=u'input_path', default=None)
    parser.add_argument('-first_rate', type=float, help=u'first fp rate', default=1.0)
    parser.add_argument('-n_row', type=int, help=u'# of use rows', default=15)
    parser.add_argument('-ymax', type=float, help=u'ylim max', default=0.7)
    parser.add_argument('-food_idx', type=int, help=u'# of use rows', default=1)

    params = parser.parse_args()

    return vars(params)

def fp_train(output_dir, input_path, first_rate, n_row, ymax, food_idx):
    df = pd.read_csv(input_path, delimiter=",", header=None)
    ite = df.ix[:,0].as_matrix()[:n_row] 
    fp_rate = df.ix[:,1+(food_idx-1)*4].as_matrix()[:n_row] 

    plt.figure(figsize=(10,5))
    plt.plot(ite,fp_rate)
    plt.xticks(range(1,len(ite)+1))
    plt.yticks([0,first_rate], [0,"%.3f"%first_rate])
    plt.hlines(first_rate, xmin=0, xmax=ite[-1] + 1, linestyles='dashed', color='k')
    for x, y in zip(ite, fp_rate): 
        plt.text(x, y, "%.3f"%y, ha='center', va='top')
    plt.xlim(0.5, ite[-1] + 0.5)
    plt.ylim(0, ymax)
    plt.savefig(os.path.join(output_dir, "fp_rate.pdf"))
    plt.clf()

if __name__ == '__main__':
    params = parse()
    fp_train(**params)

