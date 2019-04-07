# _*_ coding: UTF-8 _*_
import re
import random
import numpy as np
import bayes as by

def textparse(bigstring):
    listoftokens = re.split(r'\W+', bigstring)
    relist = [token.lower() for token in listoftokens if len(token) > 2]
    #print(relist)
    return relist

def spamtest():
    doclist = []
    classlist = []
    for i in range(1, 26):
        with open('email/spam/%d.txt' % i, 'r') as fr:
            try:
                doclist.append(textparse(fr.read()))
                classlist.append(1)
            except:
                pass

        with open('email/ham/%d.txt' % i, 'r') as fr:
            try:
                doclist.append(textparse(fr.read()))
                classlist.append(0)
            except:
                pass

    vacablist = by.createVocabList(doclist)
    trainingset  = list(range(len(doclist)))
    testset = []
    for i in range(10):
        randindex = int(random.uniform(0, len(trainingset)))
        testset.append(trainingset[randindex])
        del trainingset[randindex]

    trainmat = []
    trainclasses = []
    for docindex in trainingset:
        trainmat.append(by.setOfWords2Vec(vacablist, doclist[docindex]))
        trainclasses.append(classlist[docindex])
    p0v, p1v, pspam = by.trainNB(np.array(trainmat), np.array(trainclasses))
    errcount = 0
    for docindex in testset:
        wordvect = by.setOfWords2Vec(vacablist, doclist[docindex])
        if by.classifyNB(np.array(wordvect), p0v, p1v, pspam) != classlist[docindex]:
            errcount += 1
            print('分类错误的测试集：', doclist[docindex])
    print('错误率：%.2f%%' % (100.0*errcount / len(testset)))

if __name__ == '__main__':
    spamtest()

