import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display, HTML
import seaborn as sns
import random
import copy
import operator
import time
from scipy import stats 
import ast
from statistics import mean, stdev
import json
import argparse
import sys
from datetime import datetime
import pickle
import math
import itertools


import xgboost 
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, plot_confusion_matrix
from inspect import signature
from scipy import interp
from pyitlib import discrete_random_variable as drv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
from skorch import NeuralNet, NeuralNetClassifier #For hyperparameter tuning in pytorch models 
from skorch import helper
import skorch
from skorch.callbacks import Callback
from skorch.dataset import Dataset as Dataset_skorch
from skorch.callbacks import EarlyStopping
from skorch.callbacks import LRScheduler
from skorch.callbacks import TrainEndCheckpoint
from skorch.callbacks import PrintLog
from skorch.callbacks import EpochScoring
from skorch.helper import SliceDataset
from skorch.utils import to_numpy
from skorch.helper import predefined_split
from numpy import argmax
from torch.nn.parameter import Parameter

#This needs to contain all optimizers that will be used so they can be properly imported
optims = {
    'torch.optim.Adam': torch.optim.Adam,
    'torch.optim.SGD': torch.optim.SGD,
    'torch.optim.Adagrad': torch.optim.Adagrad,
    'torch.optim.RMSprop': torch.optim.RMSprop,
    'torch.optim.AdamW': torch.optim.AdamW,
}
