# _*_ coding: utf-8 -*-

import os
import json

import cv2
import numpy as np

import image

class Evaluation:
    """
    public:
        tarin_img_paths: 学習に用いた画像のリスト 
        test_img_list: アノテーション済みのimg_listのデータフレーム
        tp, fp, fn, tn: 画像のリストにおけるindex
        predicts: 確率のリスト N *cls
        boxes: 物体の矩形領域(x,y,w,h)のリスト N * 4 or N *4cls 
    """
    def __init__(self, 
                 test_img_list, 
                 result_path,
                 food_idx,
                 train_img_paths = [],
                 step_dir = None,
                 n_record = 5
                 ):
        self._n_record = n_record
        self._test_img_list = test_img_list #path + annotation
        self._train_img_paths = train_img_paths 
        with open(result_path) as fin:
            self._result = json.load(fin)
        self.evaluation(food_idx)

    def evaluation(self, food_idx):
        self.paths = []
        self.predicts = []
        self.boxes = []
        self.true_labels = []

        n_record = self._n_record
        if n_record < 1:
            n_record = 1

        paths =  (self._test_img_list.ix[:, "path"]).as_matrix()
        labels = (self._test_img_list.ix[:, "__background__":]).as_matrix()
        labels = labels[:,food_idx]
        for img_path, true_label in zip(paths, labels):
            if img_path not in self._train_img_paths and img_path in self._result:
                self.paths.append(img_path)
                self.true_labels.append(true_label)
                res = self._result[img_path]
                #尤度が高い順にn_recordだけindexを取得
                predicts = res["predicts"] 
                boxes = res["boxes"] 
                idxs = [x for x,y in sorted(enumerate(predicts), key = lambda x : x[1][food_idx], reverse=True)][:n_record]
                predicts = [predicts[i][food_idx] for i in idxs]
                boxes = [boxes[i][4*food_idx:4*food_idx+4] for i in idxs]
                while len(predicts) < n_record:
                    predicts.append(0)
                    boxes.append([0,0,0,0])
                self.predicts.append(predicts)
                self.boxes.append(boxes)

        self.paths = np.array(self.paths) 
        self.predicts = np.array(self.predicts) 
        self.boxes = np.array(self.boxes) 
        self.true_labels = np.array(self.true_labels) 

    def evaluation_first(self, food_idx):
        self.paths = []
        self.predicts = []
        self.boxes = []
        self.true_labels = []

        n_record = self._n_record
        if n_record < 1:
            n_record = 1

        paths =  (self._test_img_list.ix[:, "path"]).as_matrix()
        labels = (self._test_img_list.ix[:, "__background__":]).as_matrix()
        labels = labels[:,food_idx]
        for img_path, true_label in zip(paths, labels):
            if img_path not in self._train_img_paths and img_path in self._result:
                self.paths.append(img_path)
                self.true_labels.append(true_label)
                res = self._result[img_path]
                #一番最初の尤度(FRCNNでなければ面積が一番大きい=画像そのまま入力した時の尤度)
                predicts = res["predicts"] 
                boxes = res["boxes"] 
                predicts = [predicts[0][food_idx]]
                boxes = [boxes[0][4*food_idx:4*food_idx+4]]
                while len(predicts) < n_record:
                    predicts.append(0)
                    boxes.append([0,0,0,0])
                self.predicts.append(predicts)
                self.boxes.append(boxes)

        self.paths = np.array(self.paths) 
        self.predicts = np.array(self.predicts) 
        self.boxes = np.array(self.boxes) 
        self.true_labels = np.array(self.true_labels) 

    def set_true_false(self, th):
        positive_idxs = (self.predicts[:, 0] >= th)
        negative_idxs = ~positive_idxs
        O_idxs = (self.true_labels == 1)
        X_idxs = ~O_idxs
        self.tp = np.where(positive_idxs & O_idxs)[0]
        self.fp = np.where(positive_idxs & X_idxs)[0]
        self.fn = np.where(negative_idxs & O_idxs)[0]
        self.tn = np.where(negative_idxs & X_idxs)[0]

    def get_tp_fp_rate(self):
        tp_rate = len(self.tp) / float(len(self.tp) + len(self.fn))
        fp_rate = len(self.fp) / float(len(self.fp) + len(self.tn))
        return tp_rate, fp_rate

    def get_prec_recall(self):
        prec = len(self.tp) / float(len(self.tp) + len(self.fp))
        recall = len(self.tp) / float(len(self.tp) + len(self.fn))
        return prec, recall

    def save_image(self, output_dir, colors = [(128,255,128), (128,255,255), (255,128,255), (255,255,128), (0,255,0), (0,255,255), (255,0,255), (255,255,0)], step_dir=None):
        R = 224
        O_idxs = (self.true_labels == 1)
        O_dir = os.path.join(output_dir, "O")
        if not os.path.exists(O_dir):
            os.mkdir(O_dir)
        X_dir = os.path.join(output_dir, "X")
        if not os.path.exists(X_dir):
            os.mkdir(X_dir)
        for idx, img_path in enumerate(self.paths):
            if step_dir != None:
                filename = os.path.basename(img_path)
                dir_no = os.path.basename(os.path.dirname(img_path))
                img_path = os.path.join(step_dir, dir_no, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_h, img_w, _ = img.shape
                img = cv2.resize(img, (R, R))
                #尤度大きいものを最後に描画
                boxes = np.array([image.convert_box(box, (img_w, img_h), (R,R)) for box in self.boxes[idx][::-1] ])
                predicts = np.array([p for p in self.predicts[idx][::-1]])
                boxes = boxes[predicts != 0]
                predicts = predicts[predicts != 0]
                predicts = np.array(["%.3f"%p for p in predicts])
                img = image.rectangle(img, boxes,labels=predicts, colors=colors, linewidth=1, fontsize=0.8)
                if O_idxs[idx]:
                    save_dir = O_dir
                else:
                    save_dir = X_dir
                cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)

    def save_image_raw_and_rect(self, output_dir, food_idx, colors = [(0,255,0), (0,255,255), (255,0,255), (255,255,0), (128,255,128), (128,255,255), (255,128,255), (255,255,128)], step_dir=None):
        R = 224
        raw_dir = os.path.join(output_dir, "raw")
        if not os.path.exists(raw_dir):
            os.mkdir(raw_dir)
        rect_dir = os.path.join(output_dir, "rect")
        if not os.path.exists(rect_dir):
            os.mkdir(rect_dir)

        for idx, img_path in enumerate(self.paths):
            res = self._result[img_path]
            if step_dir != None:
                filename = os.path.basename(img_path)
                dir_no = os.path.basename(os.path.dirname(img_path))
                img_path = os.path.join(step_dir, dir_no, filename)
            if os.path.exists(img_path):
                boxes = res["boxes"] 
                boxes = [b[4*food_idx:4*food_idx+4] for b in boxes]
                img = cv2.imread(img_path)
                img_h, img_w, _ = img.shape
                img = cv2.resize(img, (R, R))
                cv2.imwrite(os.path.join(raw_dir, os.path.basename(img_path)), img)

                boxes = np.array([image.convert_box(box, (img_w, img_h), (R,R)) for box in boxes])
                img = image.rectangle(img, boxes, colors=colors, linewidth=1)
                cv2.imwrite(os.path.join(rect_dir, os.path.basename(img_path)), img)

    def save_image_require(self, output_dir, food_idx, step_dir=None, req=0.2):
        R = 224
        O_idxs = (self.true_labels == 1)
        O_dir = os.path.join(output_dir, "O_req")
        if not os.path.exists(O_dir):
            os.mkdir(O_dir)
        X_dir = os.path.join(output_dir, "X_req")
        if not os.path.exists(X_dir):
            os.mkdir(X_dir)
        
        tp_rates, fp_rates = self.get_roc_curve(n_point = len(O_idxs))
        tp_rates = np.sort(np.array(tp_rates))
        fp_rates = np.sort(np.array(fp_rates))
        req_ids = (fp_rates >= 0.2)
        top_rank = len(np.where(req_ids)[0]) #上位何件をPositiveとするか

        sort_idxs = np.argsort(self.predicts[:, 0])[::-1]
        pos_idxs = sort_idxs[:top_rank]

        for idx, img_path in enumerate(self.paths):
            if step_dir != None:
                filename = os.path.basename(img_path)
                dir_no = os.path.basename(os.path.dirname(img_path))
                img_path = os.path.join(step_dir, dir_no, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_h, img_w, _ = img.shape
                img = cv2.resize(img, (R, R))
                #尤度最高のみ描画
                boxes = np.array([image.convert_box(self.boxes[idx][0], (img_w, img_h), (R,R))])
                #positiveなら緑 negativeなら赤
                if idx in pos_idxs:
                    colors = [(0,255,0)]
                else:
                    colors = [(0,0,255)]
                img = image.rectangle(img, boxes, colors=colors, linewidth=1)

                if O_idxs[idx]:
                    save_dir = O_dir
                else:
                    save_dir = X_dir
                cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)

    def save_image_idxs(self, output_dir, food_idx, idxs, colors, step_dir=None):
        R = 224
        for idx in idxs:
            img_path = self.paths[idx]
            if step_dir != None:
                filename = os.path.basename(img_path)
                dir_no = os.path.basename(os.path.dirname(img_path))
                img_path = os.path.join(step_dir, dir_no, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_h, img_w, _ = img.shape
                img = cv2.resize(img, (R, R))
                #尤度最高のみ描画
                boxes = np.array([image.convert_box(self.boxes[idx][0], (img_w, img_h), (R,R))])
                #predict = "%.3f" % self.predicts[idx][0]
                #img = image.rectangle(img, boxes, labels=[predict], colors=colors, linewidth=1)
                img = image.rectangle(img, boxes, colors=colors, linewidth=1)
                cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)
                

    def get_image_req_idx(self, food_idx, step_dir=None, req=0.2):
        O_idxs = (self.true_labels == 1)
        
        tp_rates, fp_rates = self.get_roc_curve(n_point = len(O_idxs))
        tp_rates = np.sort(np.array(tp_rates))
        fp_rates = np.sort(np.array(fp_rates))
        req_ids = (fp_rates < req)
        top_rank = len(np.where(req_ids)[0]) #上位何件をPositiveとするか
        print top_rank

        sort_idxs = np.argsort(self.predicts[:, 0])[::-1]
        pos_idxs = sort_idxs[:top_rank]

        result = [ [[], []], [[], []] ] # O(pos, neg) X(pos, neg)
        for idx, img_path in enumerate(self.paths):
            if step_dir != None:
                filename = os.path.basename(img_path)
                dir_no = os.path.basename(os.path.dirname(img_path))
                img_path = os.path.join(step_dir, dir_no, filename)
            if os.path.exists(img_path):
                if O_idxs[idx]:
                    i = 0 
                else:
                    i = 1 

                if idx in pos_idxs:
                    j = 0 
                else:
                    j = 1 
                result[i][j].append(idx)
        return result


    def save_image_rank(self, output_dir, colors = [(0,255,0), (0,255,255), (255,0,255), (255,255,0), (128,255,128), (128,255,255), (255,128,255), (255,255,128)], step_dir=None):
        R = 224
        O_idxs = (self.true_labels == 1)
        O_dir = os.path.join(output_dir, "O_rank")
        if not os.path.exists(O_dir):
            os.mkdir(O_dir)
        X_dir = os.path.join(output_dir, "X_rank")
        if not os.path.exists(X_dir):
            os.mkdir(X_dir)

        sort_idxs = np.argsort(self.predicts[:, 0])

        for idx, img_path in enumerate(self.paths[sort_idxs]):
            if step_dir != None:
                filename = os.path.basename(img_path)
                dir_no = os.path.basename(os.path.dirname(img_path))
                img_path = os.path.join(step_dir, dir_no, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_h, img_w, _ = img.shape
                img = cv2.resize(img, (R, R))
                #尤度大きいものを最後に描画
                boxes = np.array([image.convert_box(box, (img_w, img_h), (R,R)) for box in self.boxes[sort_idxs][idx][::-1] ])
                predicts = np.array([p for p in self.predicts[sort_idxs][idx][::-1]])
                boxes = boxes[predicts != 0]
                predicts = predicts[predicts != 0]
                img = image.rectangle(img, boxes, colors=colors, linewidth=1)
                if O_idxs[sort_idxs][idx]:
                    save_dir = O_dir
                else:
                    save_dir = X_dir
                cv2.imwrite(os.path.join(save_dir, ("%06d_"%idx)+os.path.basename(img_path)), img)

    def get_roc_curve(self, n_point = None):
        tp_rates = []
        fp_rates = []
        sort_idxs = np.argsort(self.predicts[:, 0])
        O_idxs = (self.true_labels == 1)
        X_idxs = ~O_idxs
        if n_point is None:
            n_point = len(sort_idxs)
        for i in list(range(n_point)):
            positive_idxs = np.array([False] * len(sort_idxs))
            positive_idxs[sort_idxs[int(len(sort_idxs) * i / n_point):]] = True
            negative_idxs = ~positive_idxs
            tp = np.where(positive_idxs & O_idxs)[0]
            fp = np.where(positive_idxs & X_idxs)[0]
            fn = np.where(negative_idxs & O_idxs)[0]
            tn = np.where(negative_idxs & X_idxs)[0]
            tp_rate = len(tp) / float(len(tp) + len(fn))
            fp_rate = len(fp) / float(len(fp) + len(tn))
            tp_rates.append(tp_rate)
            fp_rates.append(fp_rate)
        tp_rates.append(0)
        fp_rates.append(0)
        return tp_rates, fp_rates

    def get_ap(self):
        #average precision
        precs = []
        sort_idxs = np.argsort(self.predicts[:, 0])
        O_idxs = (self.true_labels == 1)
        #X_idxs = ~O_idxs
        O_idxs = O_idxs[sort_idxs][::-1] #尤度高い順
        for i, is_true in enumerate(O_idxs):
            if is_true:
                precs.append(len(np.where(O_idxs[:i+1])[0]) / float(i+1))
        return np.mean(precs)

    def get_mean_predict(self):
        return np.mean(self.predicts[:, 0])
