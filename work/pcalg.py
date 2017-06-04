import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Data import Data
import numpy as np
from sklearn.svm import SVC

#color=['b','r','g','y','c','m','k']
color=[(0,0,0.9),(0.9,0,0),(0,0.4,1),(1,0.4,0),(0,0.9,1),(1,0.9,0),(0,0,0)]

def draw(p,vecs,name,label):
    x=[]
    y=[]
    for vec in vecs:
        x.append(vec[0])
        y.append(vec[1])
    fig=plt.subplot(p)
    for j in range(0,7):
        xx=[]
        yy=[]
        for i in range(0,len(x)):
            if label[i]==j:
                xx.append(x[i])
                yy.append(y[i])
        plt.scatter(xx,yy,color=color[j])
    fig.set_title(name)

def cmpp(x,y):
    a=x['d']
    b=y['d']
    if a<b:
        return -1
    if a>b:
        return 1
    return 0

def pca(name,ker):
    data=Data(name)
    data.readData()
    pca=PCA(n_components=2)
    vecs=[]
    labels=[]
    labels2=[]
    vecs.extend(data.feature_train)
    vecs.extend(data.feature_validation)
    vecs.extend(data.test)
    for i in data.label_train:
        labels.append(int(i+1e-6))
        labels2.append(i)
    for i in data.label_validation:
        labels.append(int(i+1e-6)+2)
        labels2.append(i)
    for i in range(0,len(data.test)):
        labels.append(6)
    vecs_pca=pca.fit_transform(vecs)
    l2=len(data.test)
    l1=len(vecs)-l2
    tot=tot0=tot1=0
    for j in range(0,l2):
        d=[]
        for i in range(0,l1):
            ll=(vecs_pca[i][0]-vecs_pca[j+l1][0])**2+(vecs_pca[i][1]-vecs_pca[j+l1][1])**2
            if ll<100:
                d.append({'tar':i,'d':ll})
        d.sort(cmp=cmpp)
        s0=s1=0.0
        t0=t1=0
        for i in range(0,min(10,len(d))):
            p=labels[d[i]['tar']]
            s=d[i]['d']
            if p%2==0:
                s0+=1.0/(s+0.05)
                t0+=1
            else:
                s1+=1.0/(s+0.05)
                t1+=1
        s0*=t0
        s1*=t1
        if s0/(s0+s1)<0.7 and s1/(s0+s1)<0.7:
            tot+=1
        if s0/(s0+s1)>=0.7:
            tot0+=1
            labels[l1+j]=4
        if s1/(s0+s1)>=0.7:
            tot1+=1
            labels[l1+j]=5
    print tot,tot0,tot1,l2
    plt.figure(1,figsize=(20,10))
    draw(121,vecs_pca,'PCA',labels)
    clf=SVC(kernel=ker,degree=2,C=1)
    clf.fit(vecs_pca[:l1],labels2)
    res=clf.predict(vecs_pca[-l2:])
    tot0=tot1=0
    f=file('./luogan/'+name+'.txt','w')
    for i in range(0,l2):
        if res[i]<1e-3:
            tot0+=1
            f.write('0\n')
        else:
            tot1+=1
            f.write('1\n')
        labels[-l2+i]=res[i]+4
    f.close()
    print tot0,tot1
    draw(122,vecs_pca,'SVM',labels)
    plt.show()

if __name__ == "__main__":
    pca('bird','rbf')
#    pca('cat','rbf')
