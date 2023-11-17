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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str, help="model, choices = {svm ,Dtree}",default="svm,Dtree", )
parser.add_argument("--test_size",type=float, help="test_size",default=0.2)
parser.add_argument("--dev_size",type=float, help="dev_size",default=0.2)
parser.add_argument("--runs",type=int, help="runs",default=2)
args = parser.parse_args()

digits = datasets.load_digits()

X, y = preprocess(digits)
for i, j in zip(X,y):
    if y ==1:
        print(i,j)
        break

print("Total number of samples: ",len(X))

# Create Train_test_dev size groups
# test_sizes = [0.1, 0.2, 0.3] 
# dev_sizes  = [0.1, 0.2, 0.3]
test_sizes = [args.test_size] 
dev_sizes  = [args.dev_size]
test_dev_size_groups = [{"test_size":i, "dev_size":j} for i in test_sizes for j in dev_sizes] 

# Create a classifier: a support vector classifier
# model = svm.SVC
# models = ["svm","Dtree"]
models_ = args.model.split(",")
models =[ i for i in models_ if i != ","]
print(models)
logs = compare_models(models,X,y,test_dev_size_groups,runs=args.runs,logs=True)
print("Confusion Matrix between production and Candidate mpdels on test dataset")
production_predictions = logs['svm']["_test_predicted"]
candidate_productions = logs['Dtree']["_test_predicted"]
print(metrics.confusion_matrix(production_predictions,candidate_productions))
print("Confusion matrix 2x2:")
print(metrics.confusion_matrix(production_predictions==candidate_productions,production_predictions==candidate_productions))

