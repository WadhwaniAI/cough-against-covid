"""Defines the factory object for all the initializers"""
from torch.nn.init import kaiming_uniform_, kaiming_normal_, ones_, zeros_, \
	normal_, constant_, xavier_uniform_, xavier_normal_
from cac.factory import Factory

factory = Factory()
factory.register_builder('kaiming_uniform', kaiming_uniform_)
factory.register_builder('kaiming_normal', kaiming_normal_)
factory.register_builder('ones', ones_)
factory.register_builder('zeros', zeros_)
factory.register_builder('normal', normal_)
factory.register_builder('constant', constant_)
factory.register_builder('xavier_uniform', xavier_uniform_)
factory.register_builder('xavier_normal', xavier_normal_)
