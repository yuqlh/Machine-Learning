import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN
from knn_test2 import classify0 as kNN2

def img2vector(filename):

    revector = np.zeros((1, 1024))
    f = open(filename)

    for i in range(32):
        line = f.readline()
        for j in range(32):
            revector[0, i*32 + j] = int(line[j])
    f.close()

    return revector

def createDataSet(dirname):

    labels = []
    trainlist = listdir(dirname)

    rows = len(trainlist)
    trainmat = np.zeros((rows, 1024))

    for i in range(rows):
        filename = trainlist[i]
        num = int(filename.split('_')[0])
        labels.append(num)
        trainmat[i, :] = img2vector(dirname + '/' + filename)

    return trainmat, labels

def handwritingClassTest():

    traindir = 'trainingDigits'
    trainmat, trainlabels = createDataSet(traindir)

    testdir = 'testDigits'
    testmat, testlabels = createDataSet(testdir)

    knn = kNN(n_neighbors=3, algorithm='auto')
    knn.fit(trainmat, trainlabels)

    rows = testmat.shape[0]
    errcount = 0
    testlist = listdir(testdir)
    for testfile in testlist:
        testlabel = int(testfile.split('_')[0])
        testvector = img2vector(testdir + '/' + testfile)
        re = knn.predict(testvector)
        #re = kNN2(testvector, trainmat, trainlabels, 3)
        if re!=testlabel:
            errcount += 1
    print('总个数：%d，错误个数：%d，错误率%f%%' % (rows, errcount, 100.0*errcount/rows))


if __name__ == '__main__':
    handwritingClassTest()
