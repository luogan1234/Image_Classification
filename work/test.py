#coding:UTF-8
from Data import Data
from sklearn.ensemble import RandomForestClassifier

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
    print o,i
    return res.predict(data.test)

if __name__ == '__main__':
    RandomForest('bird')
    RandomForest('cat')
