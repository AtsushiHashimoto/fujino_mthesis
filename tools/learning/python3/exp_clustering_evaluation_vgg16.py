# _*_ coding: utf-8 -*-
# Python 3.x

import os
import sys
import json
import argparse

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import isolated_dense_clustering as idc
from collections import Counter
from itertools import combinations

from sklearn.metrics.pairwise import pairwise_kernels

sys.path.append(os.path.join(__file__, "../"))
from module import image
from module.ExecutionTime import ExecutionTime

CLUSTER = "0002"
DIR_SUF = "_REG3_P1N4"

MN = 11 
MX = 30 

METRIC = "linear"
LABEL_DIR = "/home/fujino/work/output/FlowGraph/label/"
ITER = 0
INPUT_DIR = "/home/fujino/work/hdf5/%s/refine%s/iter%d/" % (CLUSTER, DIR_SUF, ITER)
#OUTPUT_DIR = "/home/fujino/work/hdf5/%s/refine%s/iter%d/fp_estimation/" % (CLUSTER, DIR_SUF, ITER)
OUTPUT_DIR = "/home/fujino/work/output/fp_estimation/zero/0002_3_10" 
IMG_LIST_PATHS =[
"/home/fujino/work/hdf5/%s/train_annotation/img_list_train%04d_annotation.tsv" % (CLUSTER, i) for i in range(100)
]
IMG_LIST_PATHS =[
path for path in IMG_LIST_PATHS if os.path.exists(path)
]


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', help=u'input directory which contains features.npy',
                        default = INPUT_DIR)
    parser.add_argument('-output_dir', help=u'output directory',
                        default = OUTPUT_DIR)
    parser.add_argument('-label_dir', help=u'label directory',
                        default = LABEL_DIR)
    parser.add_argument('-img_list_paths', help=u'img_list.tsv', action="append",
                        default = IMG_LIST_PATHS)
    parser.add_argument('-step_dir', help=u'image directory',
                        default = None)
    parser.add_argument('-metric', help=u'metric of similarities',
                        default = METRIC)

    params = parser.parse_args()

    return vars(params)


def convert_path(step_dir, img_path):
    if step_dir != None:
        filename = os.path.basename(img_path)
        dir_no = os.path.basename(os.path.dirname(img_path))
        img_path = os.path.join(step_dir, dir_no, filename)
    return img_path


def load_labels(input_path, NEtype_list=["F"], use_ontology=True, step=None):
    """
        input_path: generate_labelsの出力結果
        NEtype_list: 使用するNEtypeのリスト
        use_ontology: オントロジーを使う:True 使わない:False
    """
    with open(input_path, 'r') as fin:
        img_label = json.load(fin)

    labels = []
    for node in img_label.values():
        if node["NEtype"] in NEtype_list:
            if step is None or node["step"] == step:
                if use_ontology:
                    if node["ontology"] != None:
                        labels.append(node["ontology"])
                else:
                    labels.append(node["rNE"])

    labels = list(set(labels)) #重複を除去

    return labels


def change_same_image_score(matrix, imgs):
    paths = imgs[:, 0]
    print ("n_zero", len(np.where(matrix == 0)[0]))
    for row, path in enumerate(paths):
        cols = (paths == path)
        cols[row] = False #対角成分
        matrix[row, cols] = 0
    print ("n_zero", len(np.where(matrix == 0)[0]))
    return matrix


def estimate_false_positive(features, imgs, metric, min_k=2, max_k=10):
    """
        featuresの中で外れ値となるindexを返す
    """

    ex_time = ExecutionTime()
    MIN_K = min_k 
    MAX_K = max_k 
    search_range=range(MIN_K, MAX_K+1)
    sim_matrix = pairwise_kernels(features, metric=metric)
    #sim_matrix = change_same_image_score(sim_matrix, imgs)
    model = idc.IsolatedDenseClustering(search_range=search_range,
                                    affinity='precomputed',
                                    assign_labels='discretize',
                                    n_jobs=1,
                                    eigen_solver='arpack',
                                    random_state=0
                                    )
    print ("spectral clustering...")
    ex_time.start()
    scores, labels_all = model.fit_all(sim_matrix)
    ex_time.end()

    print ("score")
    print (scores)
    n_cluster_idx = np.argmax(scores)  
    print ("# of cluster ", search_range[n_cluster_idx])

    return n_cluster_idx, search_range, labels_all


def outlier_cluster_with_probs(probs, labels_all, food_idx):
    #各クラスタ数に対して
    mean_probs = []
    cls_no = sorted(list(set(labels_all)))
    for c in cls_no:
        mean_prob = np.mean(probs[labels_all == c, food_idx])
        mean_probs.append(mean_prob)
    #各クラスタのmed_probの最小をoutlier_clusterとする
    out_cls_idx = np.argmin(mean_probs) 
    out_cls = cls_no[out_cls_idx] 

    print ("mean_probs", mean_probs)
    print ("out", out_cls)

    return out_cls, mean_probs[out_cls]


def outlier_cluster_with_sims(features, labels_all, metric):
    #各クラスタ数に対して
    msims = []
    cls_no = sorted(list(set(labels_all)))
    for c in cls_no:
        msim = mean_sims(features[labels_all == c, :], metric)
        #msim = nn_sims(features[labels_all == c, :], metric)
        msims.append(msim)
    out_cls_idx = np.argmin(msims) 
    out_cls = cls_no[out_cls_idx] 

    print ("mean_sims", msims)
    print ("out", out_cls)

    return out_cls, msims[out_cls]


def evaluation(imgs, img_list_paths, probs, clustering_result, food_idx, label_dir, out_cls=0, step_dir=None, save=False, output_dir=None):
    def _Fmeasure(prec, recall):
        if prec == 0 and recall == 0:
            return 0
        else:
            return 2 * prec * recall / float(prec + recall)

    
    mx_cls_no = np.max(clustering_result) 
    cls_no = list(range(mx_cls_no + 1))
    result = [[0, 0]  for c in cls_no] # 各クラスタで食材が実際に写っていなかったものと写っていたものの数
    df = pd.concat([pd.read_csv(img_list_path, delimiter="\t", encoding="utf-8") for img_list_path in img_list_paths])
    food = df.columns[8+food_idx]

    probs_rank = np.argsort(probs[:, food_idx])
    p_prob = [[], []] #TP, FN  
    n_prob = [[], []] #FP, TN
    text_flow = [[0,0], [0,0], [0,0], [0,0]]  #TP FN FP TN
    for row in df.as_matrix():
        img_path, resize_w, resize_h, x, y, w, h = row[:7] 
        true_labels = row[8:]
        idx = np.where(np.all(imgs==row[:7],axis=1))[0] #一つだけのはず 
        if len(idx) > 0:
            idx = idx[0] #一つだけのはず
            cls = clustering_result[idx]
            tl = int(true_labels[food_idx]) 
            result[cls][tl] += 1
            
            if tl == 0:
                if cls == out_cls:
                    i = 1 
                else:
                    i = 0
                n_prob[i].append(probs[idx, food_idx])
            else:
                if cls == out_cls:
                    i = 1
                else:
                    i = 0
                p_prob[i].append(probs[idx, food_idx])
        
            root = os.path.splitext(os.path.basename(img_path))[0]
            img_dir_no = os.path.basename(os.path.dirname(img_path))
            label_path = os.path.join(label_dir, img_dir_no, '%s.json'%root)
            stp = int((os.path.splitext(os.path.basename(img_path))[0]).split("_")[-1])
            stp_labels=load_labels(label_path, NEtype_list=["F"], use_ontology=True, step=stp)

            flow = 0
            if food in stp_labels: #手順文章から直接
                flow = 1

            if tl == 1:
                if cls != out_cls:
                    i = 0 #TP
                else:
                    i = 1 #FN
            else:
                if cls != out_cls:
                    i = 2 #FP
                else:
                    i = 3 #TN
            text_flow[i][flow] += 1


            #save
            if save:
                img_path = convert_path(step_dir, img_path)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (int(resize_w), int(resize_h)))
                img = image.rectangle(img, [(x,y,w,h)])

                if cls == out_cls:
                    save_dir = os.path.join(output_dir, "cluster_%03d_out"%cls)
                else:
                    save_dir = os.path.join(output_dir, "cluster_%03d"%cls)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                if flow == 0:
                    save_dir = os.path.join(save_dir, "flow")
                else:
                    save_dir = os.path.join(save_dir, "text")
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                save_dir = os.path.join(save_dir, "label_%03d"%tl)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                img_path = os.path.splitext(os.path.basename(img_path))
                cv2.imwrite(os.path.join(save_dir, 
                                        "%06d_"%np.where(probs_rank == idx)[0][0] + 
                                        "%.2e_"%probs[idx,food_idx] + 
                                        img_path[0] + 
                                        "_%d_%d_%d_%d"%(x,y,w,h) + 
                                        img_path[1]), img)

    plt.figure(figsize=(12, 8))
    n_stack = 0
    p_stack = 0
    for i, v in enumerate(zip(result, cls_no)):
        res, c = v
        n, p = res 
        plt.bar([0,1], [n,p], bottom=[n_stack, p_stack], label="%d"%c, color=cm.rainbow(float(i) / len(cls_no)))
        n_stack += n
        p_stack += p
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "each_cls_pn.pdf"))
    plt.clf()

    test_images = np.sum(result) 

    if float(np.sum(result[0])) > 0:
        prec = result[out_cls][0] / float(np.sum(result[out_cls])) 
    else:
        prec = 0
    if float(np.sum([r[0] for r in result])) > 0:
        recall = result[out_cls][0] / float(np.sum([r[0] for r in result])) 
    else:
        recall = 0
    f = _Fmeasure(prec, recall) 

    #noise_rate
    if float(np.sum([np.sum(r) for r in result])) > 0:
        pre_noise = float(np.sum([r[0] for r in result])) / float(np.sum([np.sum(r) for r in result]))
    else:
        pre_noise = 0
    pre_true = float(np.sum([r[1] for r in result]))
    result[out_cls] = [0,0] # out_clsを0にする
    if float(np.sum([np.sum(r) for r in result])) > 0:
        post_noise = float(np.sum([r[0] for r in result])) / float(np.sum([np.sum(r) for r in result]))
    else:
        post_noise = 0
    rest_true = float(np.sum([r[1] for r in result]))

    #true_rate
    if pre_true > 0:
        rest_true_rate = rest_true / pre_true
    else:
        rest_true_rate = 0

    print (len(cls_no))
    print (Counter(clustering_result))
    print ("test images: %d"%(test_images))
    print ("precision: %.3f\trecall: %.3f\tF-measure: %.3f"%(prec,recall,f))
    print ("noise_rate %.3f -> %.3f"%(pre_noise, post_noise))
    print ("true %d -> %d  %.3f"%(pre_true, rest_true, rest_true_rate))
    print ("positive %.2e  negative %.2e"%(np.mean(np.hstack(p_prob)),np.mean(np.hstack(n_prob))))

    #確率ヒストグラム
    plt.figure(figsize=(12, 8))
    plt.hist(p_prob + n_prob, bins=50, alpha=1, label=["TP", "FN", "FP", "TN"], stacked=True)
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "prob_hist.pdf"))
    plt.clf()

    fig = plt.figure(figsize=(24, 8))
    for i, prob in enumerate([p_prob, n_prob]):
        if i == 0:
            ax = fig.add_subplot(1,2,i+1)
            l = ["TP", "FN"]
        else:
            ax = fig.add_subplot(1,2,i+1, sharex=ax, sharey=ax)
            l = ["FP", "TN"]
        plt.hist(prob, bins=50, alpha=1, label=l, stacked=True)
        plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "pos_neg_prob_hist.pdf"))
    plt.clf()

    print (text_flow)
    if float(text_flow[0][0] + text_flow[1][0]) > 0:
        flow_true_rate = float(text_flow[0][0]) / float(text_flow[0][0] + text_flow[1][0])
    else:
        flow_true_rate = 0 
    if float(text_flow[0][1] + text_flow[1][1]) > 0:
        text_true_rate = float(text_flow[0][1]) / float(text_flow[0][1] + text_flow[1][1])
    else:
        text_true_rate = 0 
    print (text_flow, text_true_rate, flow_true_rate)

    result = pd.DataFrame(text_flow)
    result.to_csv(os.path.join(output_dir, "text_flow.csv"), index=False, header=False)

    return pre_noise, post_noise, rest_true_rate, text_true_rate, flow_true_rate


def nn_sims(features, metric, k=10):
    n_data = len(features)
    sim_matrix = pairwise_kernels(features, metric=metric)
    sim_matrix = sim_matrix - sim_matrix * np.identity(n_data)
    nns = []
    for i in range(k): 
        idx = np.argmax(sim_matrix, axis=1)
        if len(nns) == 0:
            nns = sim_matrix[range(n_data), idx]
        else:
            nns = np.vstack([nns, sim_matrix[range(n_data), idx]])
        sim_matrix[range(n_data), idx] = 0
    mean = np.mean(np.mean(nns, axis = 0)) #k近傍の平均の平均
    return mean

def mean_sims(features, metric):
    n_data = len(features)
    sim_matrix = pairwise_kernels(features, metric=metric)
    mean = (np.sum(sim_matrix) - np.sum(np.diag(sim_matrix))) / (n_data**2 - n_data)
    return mean

def between_clusters(a_features, b_features, metric):
    sim_matrix = pairwise_kernels(a_features, b_features, metric=metric)
    mean = np.mean(sim_matrix)
    return mean

def iso_score(features, labels, target_idx, metric):
    target_features = features[labels == target_idx, :]
    s = mean_sims(target_features, metric)
    n_cls = np.max(labels) + 1
    s -= np.mean([between_clusters(target_features, features[labels == i, :], metric) for i in range(n_cls) if i != target_idx])
    return s


def remove_low_probs(imgs, img_list_paths, probs, food_idx, label_dir, output_dir):
    ex_time = ExecutionTime()
    ex_time.start()

    fig1 = plt.figure(figsize=(10, 4))
    fig2 = plt.figure(figsize=(10, 4))
    SPLIT = 10 

    idxs = np.argsort(probs[:, food_idx])

    noise_rates = []
    rest_true_rates = []
    data_nos = []
    ths = []
    for i in range(SPLIT):
        cls_label = np.array([0] * len(probs)).astype(int)
        n_low_prob = int(float(len(idxs))/SPLIT*i)
        print (n_low_prob)
        cls_label[idxs[n_low_prob:]] = 1
        ths.append(probs[idxs[n_low_prob],food_idx])
        data_no = len(np.where(cls_label == 1)[0])
        data_nos.append(data_no)
        _, noise_rate, rest_true_rate, _, _ = evaluation(imgs, img_list_paths, probs, list(cls_label), food_idx, label_dir)
        noise_rates.append(noise_rate)
        rest_true_rates.append(rest_true_rate)

    ax = fig1.add_subplot(1, 1, 1)
    ax.scatter(data_nos, noise_rates)
    for x, y, v in zip(data_nos, noise_rates, ths): 
        ax.text(x, y, "%.1e"%v, ha='center', va='top')
    for x, y, v in zip(data_nos, noise_rates, noise_rates): 
        ax.text(x, y, "%.2f"%v, ha='center', va='bottom')
    ax.set_xticks(data_nos)
    ax.set_xlim(0, len(probs) * 1.05)

    ax = fig2.add_subplot(1, 1, 1)
    ax.scatter(data_nos, rest_true_rates)
    for x, y, v in zip(data_nos, rest_true_rates, ths): 
        ax.text(x, y, "%.1e"%v, ha='center', va='top')
    for x, y, v in zip(data_nos, rest_true_rates, rest_true_rates): 
        ax.text(x, y, "%.2f"%v, ha='center', va='bottom')
    ax.set_xticks(data_nos)
    ax.set_xlim(0, len(probs) * 1.05)

    fig1.savefig(os.path.join(output_dir, "noise_rates_%d.pdf" % food_idx))
    fig1.clf()
    fig2.savefig(os.path.join(output_dir, "rest_true_rates_%d.pdf" % food_idx))
    fig2.clf()
    ex_time.end()


def main(img_list_paths, input_dir, output_dir, step_dir, label_dir, metric):
    os.mkdir(output_dir)
    print("load...")
    imgs = np.load(os.path.join(input_dir, 'f_imgs.npy'))
    labels = np.load(os.path.join(input_dir, 'f_labels.npy')) #__background__は除く
    features = np.load(os.path.join(input_dir, 'features.npy'))
    orig_probs = np.load(os.path.join(input_dir, 'f_probs.npy'))

    result = []
    for i in range(labels.shape[1]):
        out_idxs = None

        food_idx = i + 1 #skip background
        orig_idxs = np.where(labels[:,i] == 1)[0]
        
        f = features[orig_idxs, :]
        ips = imgs[orig_idxs, :]
        probs = orig_probs[orig_idxs, :]

        print (f)
        print (probs)

        save_dir = os.path.join(output_dir, "food%03d"%food_idx)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        #remove_low_probs(ips, img_list_paths, probs, food_idx, label_dir, save_dir)

        print ("%d/%d" % (len(f), len(features)))

        out_features = None
        out_sim = [] 
        all_out_sim = []
        cls_in_sim = []
        noise = []
        iso_out_score = []
        iso_in_score = []
        prob_in = []
        prob_out = []
        data_no = [len(f)]
        rest_true = [1] 
        trest_true = [1] 
        frest_true = [1] 
        ITER = 1 
        
        for i in range(ITER):
            mn_k = MN 
            mx_k = MX 
            n_cluster_idx, search_range, labels_all = estimate_false_positive(f, ips, metric, mn_k, mx_k)

            save_iter_dir = os.path.join(save_dir, "iter%d"%(i+1)) 
            if not os.path.exists(save_iter_dir):
                os.mkdir(save_iter_dir)

            mean_probs = []
            out_clss = []
            post_noise = []
            rts = []
            trts = []
            frts = []
            for idx in range(len(search_range)):
                save = False
                #save = True 
                save_cls_dir = os.path.join(save_iter_dir, "cluster%d"%search_range[idx]) 
                #if save and not os.path.exists(save_cls_dir):
                if not os.path.exists(save_cls_dir):
                    os.mkdir(save_cls_dir)
                #outlier clusterを決定
                #out_cls, mean_prob = outlier_cluster_with_probs(probs, labels_all[idx], food_idx) #outlierをprobで決める
                out_cls, mean_prob = outlier_cluster_with_sims(f, labels_all[idx], metric) #outlierをsimilarityで決める
                pre, post, rt, trt, frt = evaluation(ips, img_list_paths, probs, labels_all[idx], food_idx, label_dir, out_cls, step_dir, save, save_cls_dir)
                print(mean_prob) 

                mean_probs.append(mean_prob)
                out_clss.append(out_cls)
                if len(noise) == 0:
                    noise.append(pre)
                post_noise.append(post)
                rts.append(rt)
                trts.append(trt)
                frts.append(frt)

            out_cls = out_clss[n_cluster_idx]
            noise.append(post_noise[n_cluster_idx])
            rest_true.append(rest_true[-1] * rts[n_cluster_idx])
            trest_true.append(trest_true[-1] * trts[n_cluster_idx])
            frest_true.append(frest_true[-1] * frts[n_cluster_idx])

            print ("n cluster", search_range[n_cluster_idx])

            # 今回のクラスタリング結果でのoutlierの類似度の平均
            out_sim.append(mean_sims(f[labels_all[n_cluster_idx] == out_cls, :], metric))
            iso_out_score.append(iso_score(f, labels_all[n_cluster_idx], out_cls, metric))
            prob_out.append(np.mean(probs[labels_all[n_cluster_idx] == out_cls, food_idx]))
            # 累積したoutlierの類似度の平均
            if out_features is None: 
                out_features = f[labels_all[n_cluster_idx] == out_cls, :]
            else:
                out_features = np.vstack([out_features, f[labels_all[n_cluster_idx] == out_cls, :]])
            all_out_sim.append(mean_sims(out_features, metric))
            # inlierの類似度の平均
            # クラスタごとのinlierの類似度の平均
            n_cls = np.max(labels_all[n_cluster_idx]) + 1
            tmp_prob = []
            tmp_sim = []
            tmp_iso = []
            plt.figure(figsize=(12, 8))
            for cls_no in range(n_cls): #outlierは飛ばす
                if cls_no != out_cls:
                    tmp_prob.append(np.mean(probs[labels_all[n_cluster_idx] == cls_no, food_idx]))
                    tmp_sim.append(mean_sims(f[labels_all[n_cluster_idx] == cls_no, :], metric))
                    tmp_iso.append(iso_score(f, labels_all[n_cluster_idx], cls_no, metric))

            prob_in.append(tmp_prob)
            cls_in_sim.append(tmp_sim)
            iso_in_score.append(tmp_iso)


            #similarityの分布
            fig = plt.figure(figsize=(12, 24))
            for cls_no in range(n_cls): 
                tmp_smatrix = pairwise_kernels(f[labels_all[n_cluster_idx] == cls_no, :], metric=metric)
                tmp_sims = [tmp_smatrix[i,j] for (i,j) in combinations(range(tmp_smatrix.shape[0]),2)]
                if cls_no == out_cls:
                    l= "%d(out)" % cls_no
                else:
                    l= "%d" % cls_no
                fig.add_subplot(3,1,1)
                plt.hist(tmp_sims, bins=50, alpha=0.2, label=l)
                fig.add_subplot(3,1,2)
                plt.hist(np.log(tmp_sims), bins=50, alpha=0.2, label=l)
                
                n_data = tmp_smatrix.shape[0]
                tmp_smatrix = tmp_smatrix - tmp_smatrix * np.identity(n_data)
                nns = []
                for i in range(10): 
                    idx = np.argmax(tmp_smatrix, axis=1)
                    if len(nns) == 0:
                        nns = tmp_smatrix[range(n_data), idx]
                    else:
                        nns = np.hstack([nns, tmp_smatrix[range(n_data), idx]])
                    tmp_smatrix[range(n_data), idx] = 0
                #mx_sims = np.max(tmp_smatrix, axis=1)
                fig.add_subplot(3,1,3)
                plt.hist(nns, bins=50, alpha=0.2, label=l)
            plt.legend(loc="best")
            plt.savefig(os.path.join(save_iter_dir, "sim_dist.pdf"))
            plt.clf()


            #確率ヒストグラム
            plt.figure(figsize=(12, 8))
            tmp_probs = []
            l = []
            for cls_no in range(n_cls):
                if cls_no == out_cls:
                    l.append("%d(out)" % cls_no)
                else:
                    l.append("%d" % cls_no)
                tmp_probs.append(probs[labels_all[n_cluster_idx] == cls_no, food_idx])
            plt.hist(tmp_probs, bins=50, alpha=1, label=l, stacked=True)
            plt.legend(loc="best")
            plt.savefig(os.path.join(save_iter_dir, "prob_hist.pdf"))
            plt.clf()

            if out_idxs == None:
                out_idxs = orig_idxs[labels_all[n_cluster_idx] == out_cls] 
            else:
                out_idxs = np.hstack([out_idxs, orig_idxs[labels_all[n_cluster_idx] == out_cls]])
            orig_idxs = orig_idxs[labels_all[n_cluster_idx] != out_cls] 
            f = f[labels_all[n_cluster_idx] != out_cls, :]
            ips = ips[labels_all[n_cluster_idx] != out_cls, :]
            probs = probs[labels_all[n_cluster_idx] != out_cls, :]
            data_no.append(len(f))
            print ("rest_true", rest_true)

        plt.figure(figsize=(12, 8))
        x = range(1, ITER+1)
        plt.scatter(x, out_sim, c="r", marker="D", label="out")
        plt.scatter(x, all_out_sim, c="m", marker="o", label="all_out")
        plt.scatter([i+1 for i in range(ITER) for i_sim in cls_in_sim[i]], 
                    [i_sim for i in range(ITER) for i_sim in cls_in_sim[i]], 
                    c="g", marker="x", label="cls_in")
        plt.legend(loc="best")
        plt.savefig(os.path.join(save_dir, "sim.pdf"))
        plt.clf()

        plt.scatter(x, prob_out, c="r", marker="D", label="out")
        plt.scatter([i+1 for i in range(ITER) for i_sim in prob_in[i]], 
                    [i_sim for i in range(ITER) for i_sim in prob_in[i]], 
                    c="g", marker="x", label="cls_in")
        #plt.scatter(range(ITER), in_sim, c="b", marker="d", label="in")
        plt.legend(loc="best")
        plt.savefig(os.path.join(save_dir, "prob.pdf"))
        plt.clf()

        plt.scatter(x, iso_out_score, c="r", marker="D", label="out")
        plt.scatter([i+1 for i in range(ITER) for i_sim in iso_in_score[i]], 
                    [i_sim for i in range(ITER) for i_sim in iso_in_score[i]], 
                    c="g", marker="x", label="cls_in")
        plt.savefig(os.path.join(save_dir, "iso_score.pdf"))
        plt.clf()

        plt.plot(range(len(noise)), noise)
        for x, y, v in zip(range(len(noise)), noise, data_no): 
            plt.text(x, y, v, ha='center', va='bottom')
        for x, y in zip(range(len(noise)), noise): 
            plt.text(x, y, "%.1e"%y, ha='center', va='top')
        plt.savefig(os.path.join(save_dir, "noise_rate.pdf"))
        plt.clf()

        plt.plot(range(len(rest_true)), rest_true)
        for x, y, v in zip(range(len(rest_true)), rest_true, data_no): 
            plt.text(x, y, v, ha='center', va='bottom')
        for x, y in zip(range(len(rest_true)), rest_true): 
            plt.text(x, y, "%.1e"%y, ha='center', va='top')
        plt.savefig(os.path.join(save_dir, "true_rate.pdf"))
        plt.clf()

        np.save(os.path.join(save_dir, "out_%d.npy" % food_idx), out_idxs)

        result += ["%.3f"%noise[-1], "%.3f"%rest_true[-1], "%.3f"%frest_true[-1], "%.3f"%trest_true[-1]]
    print (result)
    return result

if __name__ == "__main__":
    params = parse()
    main(**params)
