"""Defines Factory object to register various dimensionality reduction methods"""
from typing import Any
from cac.factory import Factory
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

factory = Factory()
factory.register_builder('PCA', PCA)
factory.register_builder('TSNE', TSNE)
