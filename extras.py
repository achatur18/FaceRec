from model_zoo import RetinaFace, Landmark, ArcFaceONNX
import cv2
import os
import os.path as osp
from pathlib import Path
from model_zoo.model_zoo import ModelRouter

class ImageCache:
    data = {}

def ins_get_image(name, to_rgb=False, use_cache=True):
    # key = (name, to_rgb)
    # if key in ImageCache.data:
    #     return ImageCache.data[key]
    images_dir = "/content"
    ext_names = ['.jpg', '.png', '.jpeg']
    image_file = None
    for ext_name in ext_names:
        _image_file = osp.join(images_dir, "%s%s"%(name, ext_name))
        print(_image_file)
        if osp.exists(_image_file):
            image_file = _image_file
            break
    assert image_file is not None, '%s not found'%name
    img = cv2.imread(image_file)
    if to_rgb:
        img = img[:,:,::-1]
    # if use_cache:
    #     ImageCache.data[key] = img
    return img

def get_default_providers():
    return ['CUDAExecutionProvider', 'CPUExecutionProvider']

def get_default_provider_options():
    return None

def get_any_model(model_file):
  model_dir="/content"
  # model_file=find_onnx_file(model_dir)
  router = ModelRouter(model_file)
  providers = get_default_providers()
  provider_options = get_default_provider_options()
  model = router.get_model(providers=providers, provider_options=provider_options)
  return model


import numpy as np
from numpy.linalg import norm as l2norm
#from easydict import EasyDict

class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

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