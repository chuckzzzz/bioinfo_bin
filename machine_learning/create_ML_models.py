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

def create_ML_models(**kwargs):
    if("lr_max_iter" in kwargs):
        lr_max_iter=kwargs["lr_max_iter"]
    else:
        lr_max_iter=2000

    sk_models={
        "native_bayes":GaussianNB(),
        "logistic_regression":LogisticRegression(max_iter = lr_max_iter),
        "decision_tree":tree.DecisionTreeClassifier(random_state = 1),
        "knn":KNeighborsClassifier(),
        "random_forest":RandomForestClassifier(random_state = 1),
        "support_vector_classifier":SVC(probability = True),
        "xgb":XGBClassifier(random_state =1)
    }
    estimators=[(name,model) for name, model in sk_models.items()]
    sk_models["voting_classifier"]= VotingClassifier(estimators=estimators,voting='soft')
    return sk_models

create_ML_models(lr_max_iter=20000)