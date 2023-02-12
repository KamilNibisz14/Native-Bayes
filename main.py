import numpy as np
import pandas as pd
import random as rd

iris = pd.read_csv('iris.csv')

class DataProcessing:
    @staticmethod
    def shuffle(x):
        for i in range(len(x)-1,0,-1):
            j = rd.randint(0,i)
            x.iloc[i],x.iloc[j]=x.iloc[j],x.iloc[i]
        return x

    @staticmethod
    def splitSet(x):
        n = int(len(x)*0.7)
        xTrain = x[:n]
        xVal = x[n:]
        return xTrain, xVal

    @staticmethod
    def normalize(x):
        values = iris.select_dtypes(exclude="object")
        columNames = values.columns.tolist()
        for column in columNames:
            data = x.loc[:,column]
            maxn = max(data)
            minn = min(data)
            for row in range(len(x)):
                newValue = ((x.at[row,column] -minn)/(maxn - minn))*(1 - 0) + 0
                x.at[row,column]=newValue
        return x


class NativeBayes:
    @staticmethod
    def mean(atr):
        return sum(atr)/len(atr)
    @staticmethod
    def stdev(atr,mean):
        tmp=0
        for i in atr:
            tmp+=(i-mean)**2
        std=tmp/len(atr)
        return np.sqrt(std)
    @staticmethod
    def gauss(atr,mean, std):
        exponent = np.exp(-(atr-mean)**2/(2*std**2))
        return 1/np.sqrt(2*np.pi*std**2)*exponent
    @staticmethod
    def determineFlower(p1,p2,p3):
        t1 = 1
        t2 = 1
        t3 = 1
        for i in range(0,len(p1)-1):
            t1 *= p1[i]
            t2 *= p1[i]
            t3 *= p1[i]

        t1 /= 3
        t2 /= 3
        t3 /= 3

        smpString = 'default'
        max1 = max(t1,t2,t3)
        if max1 == t1:
            return 'Virginica'
        if max1 == t2:
            return 'Setosa'
        if max1 == t3:
            return 'Versicolor'

    @staticmethod
    def classify(sample, irisTrain, name):
        prob=[]

        data=irisTrain[irisTrain['variety']==name]
        SL=data.loc[:,'sepal.width']
        mean1 = NativeBayes.mean(SL)
        stv1 = NativeBayes.stdev(SL,mean1)
        gauss1 = NativeBayes.gauss(sample['sepal.width'],mean1,stv1)
        prob.append(gauss1)

        SL=data.loc[:,'sepal.length']
        mean1 = NativeBayes.mean(SL)
        stv1 = NativeBayes.stdev(SL,mean1)
        gauss1 = NativeBayes.gauss(sample['sepal.length'],mean1,stv1)
        prob.append(gauss1)

        SL=data.loc[:,'petal.width']
        mean1 = NativeBayes.mean(SL)
        stv1 = NativeBayes.stdev(SL,mean1)
        gauss1 = NativeBayes.gauss(sample['petal.width'],mean1,stv1)
        prob.append(gauss1)

        SL=data.loc[:,'petal.length']
        mean1 = NativeBayes.mean(SL)
        stv1 = NativeBayes.stdev(SL,mean1)
        gauss1 = NativeBayes.gauss(sample['petal.length'],mean1,stv1)
        prob.append(gauss1)

        return prob

irisTrain,irisVal = DataProcessing.splitSet(iris)

sample = irisVal.iloc[0]
prVir=NativeBayes.classify(sample,irisTrain,'Virginica')
prSet=NativeBayes.classify(sample,irisTrain,'Setosa')
prVer=NativeBayes.classify(sample,irisTrain,'Versicolor')

print(NativeBayes.determineFlower(prVir,prSet,prVer))

