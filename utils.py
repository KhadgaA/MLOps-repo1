from sklearn.model_selection import train_test_split
# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm

def preprocess(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    return data, dataset.target

def image_grid(dataset):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

# Data preprocessing
def split_train_dev_test(X,y,test_size,dev_size):
    _ = test_size + dev_size
    X_train, _xtest, y_train, _ytest = train_test_split(
    X, y, test_size=_, shuffle=True)
    X_test, X_dev, y_test, y_dev = train_test_split(
    _xtest, _ytest, test_size=dev_size, shuffle=True)
    return X_train, X_test, X_dev , y_train, y_test, y_dev

# Predict the value of the digit on the test subset
def predict_and_eval(model, X_test, y_test,display=False):
    predicted = model.predict(X_test)
    if display: 
        ###############################################################################
        # Below we visualize the first 4 test samples and show their predicted
        # digit value in the title.a

        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, prediction in zip(axes, X_test, predicted):
            ax.set_axis_off()
            image = image.reshape(8, 8)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"Prediction: {prediction}")


        ###############################################################################
        # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
        # true digit values and the predicted digit values.

        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")

        plt.show()

        ###############################################################################
        # If the results from evaluating a classifier are stored in the form of a
        # :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
        # `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
        # as follows:


        # The ground truth and predicted lists
        y_true = []
        y_pred = []
        cm = disp.confusion_matrix

        # For each cell in the confusion matrix, add the corresponding ground truths
        # and predictions to the lists
        for gt in range(len(cm)):
            for pred in range(len(cm)):
                y_true += [gt] * cm[gt][pred]
                y_pred += [pred] * cm[gt][pred]

        print(
            "Classification report rebuilt from confusion matrix:\n"
            f"{metrics.classification_report(y_true, y_pred)}\n"
        )
    return metrics.accuracy_score(y_test,predicted),predicted

def tune_hparams(model,X_train, X_test, X_dev , y_train, y_test, y_dev,list_of_param_combination):
    best_acc = -1
    for param_group in list_of_param_combination:
        temp_model = model(**param_group)
        temp_model.fit(X_train,y_train)
        acc,_ = predict_and_eval(temp_model,X_dev,y_dev,False)
        if acc > best_acc:
            best_acc = acc
            best_model = temp_model
            optimal_param = param_group
    train_acc,_= predict_and_eval(best_model,X_train,y_train,False) 
    dev_acc,_ = predict_and_eval(best_model,X_dev,y_dev,False)
    test_acc,_test_predicted =  predict_and_eval(best_model,X_test,y_test,False)
    return train_acc, dev_acc, test_acc, optimal_param,_test_predicted
    
def get_combinations(param,values,combinations):
    new_combinations = []
    for value in values:
        for combination in combinations:
            combination[param] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparameter_combinations(param_groups):
    combinations = [{}]
    for param,values in param_groups.items():
        combinations = get_combinations(param,values,combinations)
    return combinations    
def get_models(model_name= "svm"):
    if model_name =="svm":
        from sklearn import svm
        return svm.SVC
    elif model_name == 'Dtree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier
def get_model_hparams(model_name = "svm"):
    if model_name =="svm":
        gamma = [0.001,0.01,0.1,1,10,100]
        C = [0.1,1,2,5,10]
        param_groups = {
            "gamma": gamma,
            "C": C
            }
    elif model_name =="Dtree":
        criterion = ["gini","entropy","log_loss"]
        max_depth = [5,10,15]
        min_samples_split = [2,5,10]
        param_groups = {
                        "criterion" :criterion,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split
                        }
    return param_groups 

import pandas as pd   
def print_log(logger):
    for model,model_results in logger.items():
        print(f"model: {model}:")
        _df = pd.DataFrame(model_results)
        print(_df.describe().loc[["mean","std"]])
        
def compare_models(models,X,y,test_dev_size_groups, runs = 1,logs = False):
    if logs:
        logger = {}
        logger_predictions = {}
        for model in models:
            logger[model] = {"train_acc" :[],
                             "dev_acc" : [],
                             "test_acc" : [],
                             }
            logger_predictions[model] = {}
            
            
    for test_dev_size in test_dev_size_groups:
        _ = 1 - (sum(test_dev_size.values()))
        print(f'****train_size: {_}, dev_size: {test_dev_size["dev_size"]}, test_size: {test_dev_size["test_size"]}****')
        for run in range(1,runs+1):
            print(f"Run : {run}")
            X_train, X_test, X_dev , y_train, y_test, y_dev = split_train_dev_test(X,y,**test_dev_size)
            for model in models:
                param_groups = get_model_hparams(model)
                temp_model = get_models(model)       
                param_combinations = get_hyperparameter_combinations(param_groups=param_groups)
                train_acc, dev_acc, test_acc, optimal_param,_test_predicted = tune_hparams(temp_model,X_train, X_test, X_dev , y_train, y_test, y_dev,param_combinations)
                print(f'model: {model}:  train_acc: {train_acc}, dev_acc: {dev_acc}, test_acc: {test_acc}, optimal_param: {optimal_param}')
                if logs:
                    logger[model]["train_acc"].append(train_acc)
                    logger[model]["dev_acc"].append(dev_acc)
                    logger[model]["test_acc"].append(test_acc)
                    logger_predictions[model]["_test_predicted"] = _test_predicted
                    
        if logs:
            print_log(logger)
    return logger_predictions
                    