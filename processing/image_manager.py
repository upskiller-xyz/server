from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


from .utils import TARGET_SIZE

@dataclass(frozen=True)
class ImgInput:
    inp: any
    target: any
    
    @property
    def out(self)->tuple[any, any]:
        return (self.inp, self.target)
    
@dataclass(frozen=True)
class ImgFname:
    inp: str
    target: str

    @property
    def out(self)->tuple[str, str]:
        return (self.inp, self.target)


class ImageManager:
    
    
    @classmethod
    def save(cls, img:tf.Tensor, fname:str)->bool:
      """
      Save image to disk
      :param img: image to save, tf.Tensor
      :param fname: filename to save to, str
      :return: True if successful, False otherwise
      """
      try:
        img = tf.cast(img, tf.uint8)
        tf.io.write_file(fname, tf.image.encode_png(img))
        return True
      except Exception as e:
        logging.error(f"Error saving image: {e}")
        return False
    

    @classmethod
    def get_image(cls, bytestring:str, h:int=TARGET_SIZE, w:int=TARGET_SIZE, norm:bool=True)->tf.Tensor:
      image = cls.load_image(bytestring)
      return cls._process_image(image, h, w, norm)

    @classmethod
    def load_image(cls, bytestring:str)->tf.Tensor:
        img = np.frombuffer(bytestring, dtype=np.uint8)
        img = cv2.imdecode(img, flags=1)
        return tf.cast(img[:, :, :3], tf.float32)
    

    
    
    @classmethod
    def _norm(cls, img:tf.Tensor)->tf.Tensor:
      return (img / 127.5) - 1
    
    @classmethod
    def denorm(cls, img:tf.Tensor)->tf.Tensor:
      return (img + 1) * 127.5
    
    @classmethod
    def _process_image(cls, img:tf.Tensor, h:int=TARGET_SIZE, w:int=TARGET_SIZE, norm:bool=True)->tf.Tensor:
      img = tf.cast(img[:, :, :3], tf.float32)
      img = cls._resize(img, h, w)
      if norm:
        img = cls._norm(img)
      return img

    @classmethod
    def _resize(cls, img:tf.Tensor, h:int=TARGET_SIZE, w:int=TARGET_SIZE)->tf.Tensor:
      return tf.image.resize(img, [h, w],
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)