#coding:UTF-8
from Data import Data
from sklearn.ensemble import RandomForestClassifier

def RandomForest(data,feature_train):
    rfc=RandomForestClassifier(n_estimators=1)
    rfc.fit(feature_train,data.label_train)
    s=rfc.score(feature_train,data.label_train)
    return s

def RandomForestTest(name,EST):
    data=Data(name)
    data.readData()
    train=data.feature_train
    validation=data.feature_validation
    test=data.test
    l=len(train[0])
    score=[0]*l
    for i in range(0,l):
        feature_train=[]
        for a in train:
            feature_train.append(a[i:(i+1)])
        score[i]=RandomForest(data,feature_train)
        if i%256==0:
            print i
    ss=[]
    ss.extend(score)
    ss.sort()
    base=ss[-128]
    chose=[]
    for j in range(0,l):
        if score[j]>base:
            chose.append(j)
    print chose
    o=0.0
    for j in range(10,min(512,l)):
        base=ss[-j]
        p=[]
        for i in range(0,l):
            if score[i]>base:
                p.append(i)
        feature_train=[]
        feature_validation=[]
        for a in train:
            d=[]
            for i in p:
                d.append(a[i])
            feature_train.append(d)
        for a in validation:
            d=[]
            for i in p:
                d.append(a[i])
            feature_validation.append(d)
        rfc=RandomForestClassifier(n_estimators=EST)
        rfc.fit(feature_train,data.label_train)
        s=rfc.score(feature_validation,data.label_validation)
        print base,s
        if s>o:
            o=s
            pp=j
            res=rfc
    print o,pp,ss[-pp]
    feature_test=[]
    base=ss[-pp]
    p=[]
    for i in range(0,l):
        if score[i]>base:
            p.append(i)
    for a in test:
        d=[]
        for i in p:
            d.append(a[i])
        feature_test.append(d)
    return res.predict(feature_test)

if __name__ == '__main__':
    RandomForestTest('bird',33)
    RandomForestTest('cat',25)
