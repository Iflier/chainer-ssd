#! /usr/bin/env python3

import argparse
import cv2
import numpy as np

import chainer
from chainer import serializers

import config
from ssd import SSD300
from multibox import MultiBoxEncoder
import voc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('image')
    args = parser.parse_args()

    model = SSD300(n_class=20, aspect_ratios=config.aspect_ratios)
    serializers.load_npz(args.model, model)

    multibox_encoder = MultiBoxEncoder(
        grids=model.grids,
        aspect_ratios=model.aspect_ratios,
        variance=config.variance)

    src = cv2.imread(args.image, cv2.IMREAD_COLOR)

    x = cv2.resize(src, (model.insize, model.insize)).astype(np.float32)
    x -= config.mean
    x = x.transpose(2, 0, 1)
    x = x[np.newaxis]

    loc, conf = model(chainer.Variable(x, volatile=True))
    boxes, conf = multibox_encoder.decode(loc.data[0], conf.data[0])

    img = src.copy()
    nms = multibox_encoder.non_maximum_suppression(boxes, conf, 0.45)
    for box, cls, score in nms:
        if score < 0.9:
            break
        box *= (img.shape[1::-1])

        cv2.rectangle(
            img,
            (int(box.left), int(box.top)),
            (int(box.right), int(box.bottom)),
            (0, 0, 255),
            3)

        name = voc.names[cls]
        (w, h), b = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        cv2.rectangle(
            img,
            (int(box.left), int(box.top)),
            (int(box.left + w), int(box.top + h + b)),
            (0, 0, 255),
            -1)
        cv2.putText(
            img,
            name,
            (int(box.left), int(box.top + h)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255))

    while True:
        cv2.imshow('result', img)
        if cv2.waitKey() == ord('q'):
            break
