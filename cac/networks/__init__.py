"""Defines Factory object to register various networks"""
from cac.factory import Factory
from cac.networks.nn import NeuralNetworkBuilder

factory = Factory()
factory.register_builder('neural_net', NeuralNetworkBuilder())
