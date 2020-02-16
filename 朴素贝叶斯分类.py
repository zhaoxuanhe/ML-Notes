#!/usr/bin/env python
# coding: utf-8

# In[73]:


from numpy import *
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']
                ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):  #获取包含在所有文档中出现的不重复词的列表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #vocabSet 和 set(document)取并集
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):  #vocabList是整个文档里面出现的不重复的词列表，inputSet 是文档的一行
    returnVec = [0]*len(vocabList)
    #print(returnVec)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word : %s is not in my vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix,trainCategory): #朴素贝叶斯训练器 trainCategory是每篇文档类别所构成的向量
    numTrainDocs = len(trainMatrix)  #获取到一共有多少文档
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  #所有文档中 p(侮辱语言) 概率是多少

    #p0Num = zeros(numWords)  
    p0Num = ones(numWords)   #避免其中一个是零导致最终结果也为零
    #p1Num = zeros(numWords)
    p1Num = ones(numWords)
    #p0Denom = 0.0
    p0Denom = 2.0
    #p1Denom = 0.0
    p1Denom = 2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #print('p1Num: '+str(p1Num))
    #p1Vect = p1Num / p1Denom 
    p1Vect = log(p1Num / p1Denom) #防止多个小数想乘导致下溢
    #p0Vect = p0Num / p0Denom
    p0Vect = log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    print('vec2Classify:'+str(vec2Classify))
    print('p1Vec:'+str(p1Vec))
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #ln(a*b) = lna + lnb 
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1 
    else:
        return 0
def testingNB():
    listOfPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(str(testEntry) + 'classified as :' + str(classifyNB(thisDoc,p0v,p1v,pAb)))
    
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(str(testEntry) + 'classified as :' + str(classifyNB(thisDoc,p0v,p1v,pAb)))
    


# In[74]:


testingNB()


# In[15]:


listOfPosts,listClasses = loadDataSet()
#print(listOfPosts)
myVocabList = createVocabList(listOfPosts)
print(myVocabList)
print(setOfWords2Vec(myVocabList,listOfPosts[0]))


# In[56]:


listOfPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOfPosts)
print(myVocabList)
trainMat=[]
for postinDoc in listOfPosts: #将所有文档都与词典比较，都变成数字矩阵
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))

p0v,p1v,pAb = trainNB0(trainMat,listClasses)

print(pAb)
print(p0v)
print(p1v)


# ### 例子1:使用朴素贝叶斯过滤垃圾邮件

# In[ ]:




