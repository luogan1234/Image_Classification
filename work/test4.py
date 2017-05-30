from Data import Data
from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble as Ensemble
import sklearn.linear_model as LinearModel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier    
import numpy as np
import random
import chord
from sklearn.model_selection import GridSearchCV

def main(name):
    data = Data(name)
    data.readData()
    model1 = SVC(kernel='rbf', decision_function_shape = 'ovr')
    model2 = RandomForestClassifier()
    model3 = Ensemble.GradientBoostingClassifier()
    model4 = KNeighborsClassifier()
    model = Ensemble.VotingClassifier(estimators = [('svm', model1), ('rf', model2), ('gb', model3),('kn', model4)], weights = [3,2,2,1])
    grid = GridSearchCV(estimator = model1, param_grid = {'C':[0.5, 2, 10]}, cv=5)
    grid.fit(data.feature_train, data.label_train)
    print "Best params: " , grid.best_estimator_.get_params()
    print grid.score(data.feature_validation, data.label_validation)

if __name__ == '__main__':
    main('bird')
    # main('cat')