#coding:UTF-8
import scipy.io as scio
import numpy as np
class Data(object):
    def __init__(self,name):
        self.name=name
        self.test=[]
        self.train=[]
        self.validation=[]
        self.feature_train=[]
        self.label_train=[]
        self.feature_validation=[]
        self.label_validation=[]

    def read(self,filename):
        return scio.loadmat('../data/'+filename+'.mat')[filename]

    def build(self):
        for vec in self.train:
            self.feature_train.append(vec[:-1])
            self.label_train.append(vec[-1])
        for vec in self.validation:
            self.feature_validation.append(vec[:-1])
            self.label_validation.append(vec[-1])
        
    def readData(self):
        self.test=self.read(self.name+'_test')
        self.train=self.read(self.name+'_train')
        self.validation=self.read(self.name+'_validation')
        # self.fake_positive = np.load('../data/fake_'+self.name+'_positive.npy')
        self.build()
        print 'read '+self.name+' data done.'
