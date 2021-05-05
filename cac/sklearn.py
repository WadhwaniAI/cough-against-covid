"""Defines Factory object to register various sklearn methods"""
from typing import Any
from cac.factory import Factory
from sklearn.svm import SVC as SVM
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

factory = Factory()
factory.register_builder('SVM', SVM)
factory.register_builder('GradientBoostingClassifier', GradientBoostingClassifier)
factory.register_builder('XGBClassifier', XGBClassifier)
factory.register_builder('LogisticRegression', LogisticRegression)

