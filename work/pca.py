import numpy as np
import scipy.io
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier


class Data:
    def __init__(self, name):
        self.name = name
        self.feature_train = []
        self.label_train = []
        self.feature_validation = []
        self.label_validation = []
        self.feature_test = scipy.io.loadmat('../data/' + name + '_test.mat')[name + '_test']
        train = scipy.io.loadmat('../data/' + name + '_train.mat')[name + '_train']
        validation = scipy.io.loadmat('../data/' + name + '_validation.mat')[name + '_validation']
        for vec in train:
            self.feature_train.append(vec[:-1])
            self.label_train.append(vec[-1])
        for vec in validation:
            self.feature_validation.append(vec[:-1])
            self.label_validation.append(vec[-1])
        self.trainnum = len(self.feature_train)
        self.validationnum = len(self.feature_validation)
        self.testnum = len(self.feature_test)
        self.feature_train = np.asarray(self.feature_train)
        self.feature_validation = np.asarray(self.feature_validation)
        self.feature_test = np.asarray(self.feature_test)
        self.label_train = np.asarray(self.label_train)
        self.label_validation = np.asarray(self.label_validation)
        print self.feature_train.shape
        print self.feature_validation.shape
        print self.feature_test.shape


if __name__ == '__main__':
    data = Data('bird')
    pca = PCA(n_components = None, copy = True, whiten = False)
    feature = np.vstack((data.feature_train, data.feature_validation, data.feature_test))
    print feature.shape 
    feature = pca.fit_transform(feature)
    print feature.shape
    f_train = feature[0 : data.trainnum, :]
    f_validation = feature[data.trainnum : data.trainnum + data.validationnum, :]
    f_test = feature[data.trainnum + data.validationnum : data.trainnum + data.validationnum + data.testnum, :]
    f_train = f_train[:, 1: 1000]
    f_validation = f_validation[:, 1: 1000]
    f_test = f_test[:, 1: 1000]
    print f_train.shape
    print f_validation.shape
    print f_test.shape
    #clf = svm.SVC(kernel = 'rbf', C = 1)
    #clf.fit(f_train, data.label_train)
    #score = clf.score(f_validation, data.label_validation)
    #print score
    for i in range(5, 50):
        rfc = BaggingClassifier(n_estimators = i)
        rfc.fit(f_train, data.label_train)
        score = rfc.score(f_validation, data.label_validation)
        print i, score
