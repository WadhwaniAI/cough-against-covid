"""Defines Factory object to register various datasets"""
from cac.factory import Factory
from cac.data.classification import ClassificationDatasetBuilder
from cac.data.unsupervised import UnsupervisedDatasetBuilder

factory = Factory()
factory.register_builder(
    "classification_dataset", ClassificationDatasetBuilder())
factory.register_builder("unsupervised_dataset", UnsupervisedDatasetBuilder())