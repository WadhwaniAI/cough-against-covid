"""Defines Factory object to register various datasets"""
from cac.factory import Factory
from cac.data.classification import ClassificationDatasetBuilder
from cac.data.unsupervised import UnsupervisedDatasetBuilder
from cac.data.context_classification import ContextClassificationDatasetBuilder

factory = Factory()
factory.register_builder(
    "classification_dataset", ClassificationDatasetBuilder())
factory.register_builder("unsupervised_dataset", UnsupervisedDatasetBuilder())
factory.register_builder(
    "context_classification_dataset", ContextClassificationDatasetBuilder())