"""Defines Factory object to register various models"""
from cac.factory import Factory
from cac.models.classification import ClassificationModelBuilder
from cac.models.dimensionality_reduction import DimensionalityReductionModelBuilder
from cac.models.classical import ClassicalModelBuilder
from cac.models.multi_signal_classification import MultiSignalClassificationModelBuilder

factory = Factory()
factory.register_builder(
    'classification', ClassificationModelBuilder())
factory.register_builder(
    'multi_signal_classification', MultiSignalClassificationModelBuilder())    
factory.register_builder(
    'dimensionality_reduction', DimensionalityReductionModelBuilder())
factory.register_builder(
	'classical', ClassicalModelBuilder())
