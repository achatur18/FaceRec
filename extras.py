from model_zoo import RetinaFace, Landmark, ArcFaceONNX
import cv2
import os
import os.path as osp
from pathlib import Path
from model_zoo.model_zoo import ModelRouter

class ImageCache:
    data = {}

def ins_get_image(image_file, to_rgb=False, use_cache=True):
    img = cv2.imread(image_file)
    if to_rgb:
        img = img[:,:,::-1]
    return img

def get_default_providers():
    return ['CUDAExecutionProvider', 'CPUExecutionProvider']

def get_default_provider_options():
    return None

def get_any_model(model_file):
  router = ModelRouter(model_file)
  providers = get_default_providers()
  provider_options = get_default_provider_options()
  model = router.get_model(providers=providers, provider_options=provider_options)
  return model


import numpy as np
from numpy.linalg import norm as l2norm

class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender==1 else 'F'