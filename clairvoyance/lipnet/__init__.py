import functools
import logging
import os

import attr
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

from lipnet.model import LipNet
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet

import clairvoyance.core

import pkg_resources

from lipnet.lipreading.videos import Video

class Reader(clairvoyance.core.Reader):
    def __init__(self, config):
        self._log = logging.getLogger(self.__class__.__name__)
        self._image_width = config.image_width
        self._image_height = config.image_height
        self._image_channels = config.image_channels
        self._frames = config.frames
        self._predict_greedy = config.predict_greedy
        self._predict_beam_width = config.predict_beam_width
        if config.predict_dictionary is None:
            self._predict_dictionary = pkg_resources.resource_filename(__name__, os.path.join('..','..','common','dictionaries','{}.txt'.format(config.predict_dictionary_type)))
        else:
            self._predict_dictionary = config.predict_dictionary
        if config.weight_path is None:
            self._weight_path = pkg_resources.resource_filename(__name__, os.path.join('..','..','evaluation','models','{}.h5'.format(config.weight_type)))
        else:
            self._weight_path = config.weight_path


    @functools.lru_cache(maxsize=1)
    def _decoder(self):
        return Decoder(greedy=self._predict_greedy, beam_width=self._predict_beam_width,
                       postprocessors=[labels_to_text, Spell(path=self._predict_dictionary).sentence])

    @functools.lru_cache(maxsize=1)
    def _lipnet(self, c, w, h, n, absolute_max_string_len=32, output_size=28):
        lipnet = LipNet(img_c=c, img_w=w, img_h=h, frames_n=n,
                        absolute_max_string_len=absolute_max_string_len, output_size=output_size)

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
        lipnet.model.load_weights(self._weight_path)
        return lipnet

    def warmup(self):
        self._lipnet(c=self._image_channels, w=self._image_width, h=self._image_height, n=self._frames)

    def do(self, frames):
        if K.image_data_format() == 'channels_first':
            img_c, frames_n, img_w, img_h = frames.shape
        else:
            frames_n, img_w, img_h, img_c = frames.shape

        assert (self._image_width, self._image_height, self._image_channels) == (img_w, img_h, img_c), '{} != {}'.format((self._image_width, self._image_height, self._image_channels), (img_w, img_h, img_c))

        X_data       = np.array([frames]).astype(np.float32) / 255
        input_length = np.array([len(frames)])

        y_pred         = self._lipnet(c=img_c, w=img_w, h=img_h, n=frames_n).predict(X_data)
        result         = self._decoder().decode(y_pred, input_length)[0]
        return result

    @attr.s
    class Config:
        image_width = attr.ib(default=100)
        image_height = attr.ib(default=50)
        image_channels = attr.ib(default=3)
        frames = attr.ib(default=256)
        predict_greedy = attr.ib(default=False)
        predict_beam_width = attr.ib(default=200)
        predict_dictionary_type = attr.ib(default='big')
        weight_type = attr.ib(default='overlapped-weights368')
        predict_dictionary = attr.ib(default=None)
        weight_path = attr.ib(default=None)
