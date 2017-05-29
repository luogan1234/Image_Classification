#coding:UTF-8
from Data import Data
from sklearn.ensemble import RandomForestClassifier
import random

EST=49

def RandomForest(name):
    data=Data(name)
    data.readData()
    o=0.0
    for i in range(0,1000):
        rfc=RandomForestClassifier(n_estimators=EST,random_state=random.randint(0,2147483647))
        rfc.fit(data.feature_train,data.label_train)
        s=rfc.score(data.feature_validation,data.label_validation)
        if s>o:
            o=s
            p=i
            res=rfc
        print s
        if i%100==0:
            print i
    print o,i
    return res.predict(data.test)

if __name__ == '__main__':
    RandomForest('bird')
    RandomForest('cat')
