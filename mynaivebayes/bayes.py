import numpy as np
from functools import reduce

def createDataset():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

def setOfWords2Vec(vocablist, inputset):
    revec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            revec[vocablist.index(word)] = 1

    return revec

def createVocabList(dataset):
    vocabset = set()
    for row in dataset:
        vocabset = vocabset | set(row)

    return list(vocabset)

def trainNB(trainmatix, traincategory):
    trainnums = len(trainmatix)
    trainwords = len(trainmatix[0])
    pabusive = sum(traincategory) / float(trainnums)
    p0num = np.ones(trainwords)
    p1num = np.ones(trainwords)
    p0denom = p1denom = 2.0
    for i in range(trainnums):
        if traincategory[i] == 1:
            p1num += trainmatix[i]
            p1denom += sum(trainmatix[i])
        else:
            p0num += trainmatix[i]
            p0denom += sum(trainmatix[i])

    p1vect = np.log(p1num / p1denom)
    p0vect = np.log(p0num / p0denom)
    return p0vect, p1vect, pabusive

def classifyNB(vec2classify, p0vec, p1vec, pclass):
    p1 = sum(vec2classify * p1vec) + np.log(pclass)
    p0 = sum(vec2classify * p0vec) + np.log(1-pclass)
    #print('p0=%f, p1=%f' % (p0, p1))
    if p1 > p0:
        return 1
    else:
        return 0

def test():
    postinglist, classvec = createDataset()
    vocablist = createVocabList(postinglist)
    #print(vocablist)
    trainmat = []
    for row in postinglist:
        trainmat.append(setOfWords2Vec(vocablist, row))
    p0v, p1v, pab = trainNB(np.array(trainmat), np.array(classvec))
    print('p0v=')
    print(sum(p0v))
    testentry = [['love', 'my', 'dalmation'],
                 ['stupid']]
    for entry in testentry:
        thisdoc = np.array(setOfWords2Vec(vocablist, entry))
        if classifyNB(thisdoc, p0v, p1v, pab):
            print(entry, '属于侮辱类')
        else:
            print(entry, '不属于侮辱类')


if __name__ == '__main__':
    test()
