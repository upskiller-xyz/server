from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from sklearn.cluster import KMeans

from .colorscale import ColorScale
from skimage.color import rgb2lab

@dataclass(frozen=False)
class ColorConverter:
    colorscale: ColorScale
    clustering = None
    
    
    def init(self):
        """
        Function to initialize the clustering algorithm
        :return: clustering model, KMeans
        """
        return KMeans(n_clusters=len(self.colorscale.colors_lab), init = [list(x) for x in self.colorscale.colors_lab], max_iter=1, n_init=1)
    
    def make(self, img:np.ndarray)->np.ndarray:
        """
        Function to convert the image to the colorscale values
        :param img: image to convert to colorscale values in RGB, np.array
        :return: image with converted colors in RGB, np.array
        
        """
        img = self.to_lab(img)
        colors = np.array([list(x) for x in self.colorscale.colors_lab])
        color_distance = img[:, :, np.newaxis, :] - colors.reshape(1, 1, -1, 3) 
        labels = np.linalg.norm(color_distance, axis=3).argmin(axis=2) 
        return colors[labels]
    
    def to_lab(self, img:np.ndarray)->np.ndarray:
        return cv2.cvtColor(np.array(img).astype(np.uint8)[:,:,:3], cv2.COLOR_RGB2Lab)

    def cluster(self, img:np.array)->np.array:
        """
        Function to cluster the image with the colorscale values
        :param img: image to cluster in RGB, np.array
        :return: clustered image, np.array
        """
        if not self.clustering:
            self.clustering = self.init()
        img = self.to_lab(img)
        res = self.clustering.fit(img.reshape(-1, 3))
        return res
    
    def label(self, img:np.array)->np.array:
        """
        Function to label the image with the colorscale values
        :param img: image to label in RGB, np.array
        :return: labeled image, np.array
        """
        if not self.clustering:
            self.clustering = self.init()
        res = self.cluster(img)
        return res.labels_.reshape(*img.shape[:2])


    def _make(self, img:np.array)->np.array:
        """
        Function to convert the image to the colorscale values
        :param img: image to convert to colorscale values in RGB, np.array
        :return: image with converted colors in RGB, np.array
        """
        if not self.clustering:
            self.clustering = self.init()
        res = self.cluster(img)
        out_labels = res.labels_.reshape(*img.shape[:2])
        return np.array([list(x.rgb) for x in self.colorscale.colors])[out_labels]
