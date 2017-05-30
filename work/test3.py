from Data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import random
import chord
def main(name):
    data = Data(name)
    data.readData()
    positive = np.array(data.label_train) > 0
    w = chord.DistributionWeight([np.array(data.feature_train)[positive], np.array(data.feature_validation)])
    rank = range(len(w))
    rank = sorted(rank, key = lambda x: w[x])
    limit = len(w) * 0.3
    feature = np.array(data.feature_train)[:, [rank[k] < limit for k in range(len(w))]]
    # model = RandomForestClassifier(n_estimators = 200)
    model = SVC(C = 10, kernel='rbf', decision_function_shape = 'ovr')
    model.fit(feature, data.label_train)
    feature_validation = np.array(data.feature_validation)[:, [rank[k] < limit for k in range(len(w))]]
    print model.score(feature_validation, data.label_validation)

if __name__ == '__main__':
    main('bird')
    main('cat')