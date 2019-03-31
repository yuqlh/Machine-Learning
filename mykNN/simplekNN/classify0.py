import numpy as np
import operator

def createDataSet():
    group = np.array([[1,101], [5,89], [108,5], [115,8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']

    return group, labels

def classify0(data, group, labels, k):
    groupSize = group.shape[0]
    diffMat = np.tile(data, (groupSize, 1)) - group

    sqMat = diffMat**2
    sqDis = sqMat.sum(axis=1)
    dis = sqDis ** 0.5
    sDis = dis.argsort()

    classCount = {}
    for i in range(k):
        lbValue = labels[sDis[i]]
        classCount[lbValue] = classCount.get(lbValue, 0) + 1

    sClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    data = [100, 20]
    dataClass = classify0(data, group, labels, 3)
    print(dataClass)
