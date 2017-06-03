#coding:UTF-8
from Data import Data
from sklearn.ensemble import RandomForestClassifier
import random

eps=1e-6

def cmp2(x,y):
    if x['tot']<y['tot']:
        return -1
    if x['tot']>y['tot']:
        return 1
    return 0

def RandomForest(name):
    data=Data(name)
    data.readData()
    train=[]
    vali=[]
    label=[]
    train.extend(data.feature_train)
    vali.extend(data.feature_validation)
    l1=len(train)
    l2=len(vali)
    l=len(train[0])
    num=[]
    for i in range(0,l):
        s=0
        for t in train:
            if t[i]<=eps:
                s+=1
        for v in vali:
            if v[i]<=eps:
                s+=1
        num.append({'flag':i,'tot':s})
    num.sort(cmp=cmp2)
    t=[]
    v=[]
    for i in range(0,l1):
        t.append([])
    for i in range(0,l2):
        v.append([])
    for j in range(0,l):
        print num[j]
        k=num[j]['flag']
        for i in range(0,l1):
            t[i].append(train[i][k])
        for i in range(0,l2):
            v[i].append(vali[i][k])
        rfc=RandomForestClassifier(n_estimators=30)
        rfc.fit(t,data.label_train)
        s=rfc.score(v,data.label_validation)
        print s
            
        
if __name__ == '__main__':
    RandomForest('bird')
    RandomForest('cat')
