# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import numpy as np
#from datasets.imdb import imdb

from imdb import imdb
from make_roidb import make_roidb

class food_recog(imdb):
    def __init__(self, name, img_list_paths, seed=0):
        imdb.__init__(self, name)
        self._image_set = name 
        
        self._classes, self._gt_roidb = make_roidb(img_list_paths) 
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = list((self._gt_roidb).keys())
        #np.random.seed(0)
        #np.random.shuffle(self._image_index)

        self._roidb_handler = self.gt_roidb


        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        #image_path = os.path.join(self._data_path, index + self._image_ext)
        image_path = index #面倒なのでのでpath = indexにしている
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        #This function loads/saves from/to a cache file to speed up future calls.
        """
        # 忘れそうなのでcacheは無しで
        #cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        #if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as fid:
        #        roidb = cPickle.load(fid)
        #    print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        #    return roidb

        roidb = [self._gt_roidb[index]
                    for index in self.image_index]
        #with open(cache_file, 'wb') as fid:
        #    cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        #print 'wrote gt roidb to {}'.format(cache_file)

        return roidb
        
    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        
        #with open(os.path.join(self._work_path, self._image_set + '_result.pickle'), 'wb') as fout:
        #    cPickle.dump((self._image_index, self._classes, self._class_to_ind, self._gt_roidb, all_boxes),
        #                    fout)
        #print "save ", os.path.join(self._work_path, self._image_set + '_result.pickle')

        pass

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

# if __name__ == '__main__':
#     d = datasets.imagenet('val1', '')
#     res = d.roidb
#     from IPython import embed; embed()
