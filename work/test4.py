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
    model1 = SVC()
    model2 = RandomForestClassifier(n_estimators = 100)
    model3 = Ensemble.GradientBoostingClassifier(n_estimators=100)
    model4 = KNeighborsClassifier()
    model = Ensemble.VotingClassifier(estimators = [('svm', model1), ('rf', model2), ('gb', model3),('kn', model4)], weights = [3,2,2,1])
    grid = GridSearchCV(estimator = model1, param_grid = {'C':[0.5, 2, 10]}, cv=5)
    all_feature = np.concatenate((np.array(data.feature_train, dtype=np.float32), np.array(data.feature_validation, dtype= np.float32)))
    all_label = np.concatenate((np.array(data.label_train, dtype = np.float32), np.array(data.label_validation, dtype=np.float32)))
    model3.fit(all_feature, all_label)
    # model2.fit(data.feature_train, data.label_train)
    # print "Best params: " , grid.best_estimator_.get_params()
    ans = model3.predict(data.test)
    print ans,sum(ans)
    np.save('gdbt_'+ name +'.npy',ans)

if __name__ == '__main__':
    # main('bird')
    main('bird')