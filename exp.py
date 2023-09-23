"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from utils import *

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.
###############################################################################

digits = datasets.load_digits()
# Classification
X, y = preprocess(digits)

# Create Param groups



gamma = [0.001,0.01,0.1,1,10,100]
C = [0.1,1,2,5,10]
# param_groups = [{"gamma":i, "C":j} for i in g for j in C] 

param_groups = {
    "gamma": gamma,
    "C": C
    }

param_combinations = get_hyperparameter_combinations(param_groups=param_groups)
# Create Train_test_dev size groups
test_sizes = [0.1, 0.2, 0.3] 
dev_sizes  = [0.1, 0.2, 0.3]
test_dev_size_groups = [{"test_size":i, "dev_size":j} for i in test_sizes for j in dev_sizes] 

# Create a classifier: a support vector classifier
model = svm.SVC
for test_dev_size in test_dev_size_groups:
    X_train, X_test, X_dev , y_train, y_test, y_dev = split_train_dev_test(X,y,**test_dev_size)
    train_acc, dev_acc, test_acc, optimal_param = tune_hparams(model,X_train, X_test, X_dev , y_train, y_test, y_dev,param_combinations)
    _ = 1 - (sum(test_dev_size.values()))
    print(f'train_size: {_}, dev_size: {test_dev_size["dev_size"]}, test_size: {test_dev_size["test_size"]} , train_acc: {train_acc}, dev_acc: {dev_acc}, test_acc: {test_acc}, optimal_param: {optimal_param}')


