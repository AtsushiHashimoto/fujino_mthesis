# _*_ coding: utf-8 -*-

import cv2

def rectangle(img, boxes,labels=None,
             linewidth = 2, fontsize=0.6, fontwidth=2, colors=[(0,255,0)],):
    """
        img: image
        boxes: list of (x,y,w,h)
        labels: list of label for boxes
        linewidth: line width of box
        fontsize: font size of label
        fontwidth: font width of label
        colors: box colors (順番に切り替わる)
    """
    if labels != None:
        assert len(boxes) == len(labels), "Boxes are inconsistent with labels. %d vs %d" % (len(boxes),len(labels))

    color_idx = 0
    for i, box in enumerate(boxes):
        x,y,w,h = box
        cv2.rectangle(img, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), colors[color_idx], linewidth)

        if labels != None:
            label = labels[i]
            #cv2.rectangle(img, (int(x), int(y)), (int(x) + int(fontsize + 1) * 11 * len(label), int(y) + int(fontsize + 1) * 14), colors[color_idx], -1) # fill サイズは結構適当
            cv2.rectangle(img, (int(x), int(y)), (int(x)+int((fontsize+1)*9*len(label)), int(y)+int((fontsize+1)*14)), colors[color_idx], -1) # fill サイズは結構適当
            cv2.putText(img, label, \
                (int(x), int(y)+int((fontsize+1)*12)), cv2.FONT_HERSHEY_SIMPLEX, \
                fontsize, (0, 0, 0), fontwidth)

        color_idx = (color_idx + 1) % len(colors)
    return img


def convert_box(box, shp, resize): 
    """
            box: (x,y,w,h)
            shp: (img_w, img_h) 
            resize: (resize_w, resize_h)
        return:
            box: (x,y,w,h) after resize
    """
    img_w, img_h = shp 
    resize_w, resize_h = resize
    x, y, w, h = box
    x = int((float(resize_w) / img_w) * x)
    y = int((float(resize_h) / img_h) * y)
    w = int((float(resize_w) / img_w) * w)
    h = int((float(resize_h) / img_h) * h)
    return [x,y,w,h]




