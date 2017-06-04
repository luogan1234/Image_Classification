#coding:UTF-8
from Data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def RandomForest(name):
    data=Data(name)
    data.readData()
    o=0.0
    for i in range(5,50):
        rfc=RandomForestClassifier(n_estimators=i)
        rfc.fit(data.feature_train,data.label_train)
        s=rfc.score(data.feature_validation,data.label_validation)
        if s>o:
            o=s
            p=i
            res=rfc
    print o,p
    return res.predict(data.test)

def SVM(name,ker):
    data=Data(name)
    data.readData()
    clf=SVC(kernel=ker,degree=2,C=1)
    clf.fit(data.feature_train,data.label_train)
    s=clf.score(data.feature_validation,data.label_validation)
    print s

if __name__ == '__main__':
#    RandomForest('bird')
#    RandomForest('cat')
    SVM('bird','rbf')
    SVM('cat','poly')
