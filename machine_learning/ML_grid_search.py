import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 

import pickle



def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))


def grid_search(model,param_grid,X_train,Y_train,model_name,cv=5,verbose=True,randomized=False,rand_iter=1000):
    if(randomized):
        clf = RandomizedSearchCV(model, param_distributions = param_grid, cv = 5, verbose = True,n_iter=rand_iter)
    else:
        clf = GridSearchCV(model, param_grid = param_grid, cv = 5, verbose = True)
    best_clf = clf.fit(X_train,Y_train)
    clf_performance(best_clf,model_name)
    return best_clf.best_estimator_

# pass in model dict, train data and their labels
def main(sk_models, X_train, Y_train):
    lr_grid={'max_iter' : [2000,5000,1e5],
          'penalty' : ['l1', 'l2'],
          'C' : np.logspace(-4, 4, 20),
          'solver' : ['liblinear']}
    knn_grid={'n_neighbors' : [3,5,7,9],
            'weights' : ['uniform', 'distance'],
            'algorithm' : ['auto', 'ball_tree','kd_tree'],
            'p' : [1,2]}
    svc_grid=[
            {'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],'C': [.1, 1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
            {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}
    ]
    rf_grid={'n_estimators': [100,200,300,400,500,600,700,800,900,1000],
            'criterion':['gini','entropy'],
            'bootstrap': [True],
            'max_depth': np.linspace(100,1000,15,dtype=int),
            'max_features': ['auto','sqrt', 10],
            'min_samples_leaf': [1,5,10],
            'min_samples_split': [1.0,5,10]
    }
    xgb_grid={
            'n_estimators': [20, 50, 100, 250, 500,1000],
            'colsample_bytree': [0.2, 0.5, 0.7, 0.8, 1],
            'max_depth': [2, 5, 10, 15, 20, 25, None],
            'reg_alpha': [0, 0.5, 1],
            'reg_lambda': [1, 1.5, 2],
            'subsample': [0.5,0.6,0.7, 0.8, 0.9],
            'learning_rate':[.01,0.1,0.2,0.3,0.5, 0.7, 0.9],
            'gamma':[0,.01,.1,1,10,100],
            'min_child_weight':[0,.01,0.1,1,10,100],
            'sampling_method': ['uniform', 'gradient_based']
    }
    grid_dict={
        "logistic_regression":lr_grid,
        "knn":knn_grid,
        "random_forest":rf_grid,
        "support_vector_classifier":svc_grid,
        "xgb":xgb_grid
    }
    best_estimators={}
    random_search=['xgb','random_forest']

    for name, model in sk_models.items(): 
        if name not in grid_dict:
            continue
        if(name not in random_search):
            best_lr=grid_search(model,grid_dict[name],X_train,Y_train,name)
        else:
            print("Random Grid Search!")
            best_lr=grid_search(model,grid_dict[name],X_train,Y_train,name,randomized=True)
        best_estimators[name]=best_lr
        outfile="./"+name+".pkl"
        with open(outfile, 'wb') as pickle_file:
            pickle.dump(best_lr,pickle_file,pickle.HIGHEST_PROTOCOL)
