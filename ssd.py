import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import initializers

from multibox import MultiBox


def normalize_2d(x, eps=1e-05):
    norm = F.sqrt(F.sum(F.square(x), axis=1)) + eps
    norm = F.broadcast_to(norm[:, np.newaxis], x.shape)
    return x / norm


class SSD300(chainer.Chain):

    insize = 300
    grids = (38, 19, 10, 5, 3, 1)

    def __init__(self, n_class, aspect_ratios):
        init = {
            'initialW': initializers.GlorotUniform(),
            'initial_bias': initializers.constant.Zero(),
        }
        super().__init__(
            base=L.VGG16Layers(pretrained_model=None),

            conv5_1=L.DilatedConvolution2D(None, 512, 3, pad=1, **init),
            conv5_2=L.DilatedConvolution2D(None, 512, 3, pad=1, **init),
            conv5_3=L.DilatedConvolution2D(None, 512, 3, pad=1, **init),

            conv6=L.DilatedConvolution2D(
                None, 1024, 3, pad=6, dilate=6, **init),
            conv7=L.Convolution2D(None, 1024, 1, **init),

            conv8_1=L.Convolution2D(None, 256, 1, **init),
            conv8_2=L.Convolution2D(None, 512, 3, stride=2, pad=1, **init),

            conv9_1=L.Convolution2D(None, 128, 1, **init),
            conv9_2=L.Convolution2D(None, 256, 3, stride=2, pad=1, **init),

            conv10_1=L.Convolution2D(None, 128, 1, **init),
            conv10_2=L.Convolution2D(None, 256, 3, **init),

            conv11_1=L.Convolution2D(None, 128, 1, **init),
            conv11_2=L.Convolution2D(None, 256, 3, **init),

            multibox=MultiBox(n_class, aspect_ratios=aspect_ratios, init=init),
        )
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios
        self.train = False

    def __call__(self, x, t_loc=None, t_conf=None):
        hs = list()

        h = self.base(x, layers=['conv4_3'])['conv4_3']
        hs.append(normalize_2d(h) * 20)
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 3, stride=1, pad=1)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)

        h_loc, h_conf = self.multibox(hs)

        if self.train:
            loss_loc, loss_conf = self.multibox.loss(
                h_loc, h_conf, t_loc, t_conf)
            loss = loss_loc + loss_conf
            chainer.report(
                {'loss': loss, 'loc': loss_loc, 'conf': loss_conf},
                self)
            return loss
        else:
            return h_loc, h_conf
