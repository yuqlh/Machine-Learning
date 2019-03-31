#_*_ coding: UTF-8 _*_

import numpy as np
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

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

def file2matrix(filename):
    f = open(filename)
    lines = f.readlines()
    rowCount = len(lines)
    reMat = np.zeros((rowCount, 3))
    reLabel = []

    index = 0
    for line in lines:
        line = line.strip()
        values = line.split('\t')
        reMat[index, :] = values[0:3]

        if values[-1]=='didntLike':
            reLabel.append(1)
        elif values[-1] == 'smallDoses':
            reLabel.append(2)
        elif values[-1] == 'largeDoses':
            reLabel.append(3)
        index += 1

    return reMat, reLabel

def autoNorm(dataMat):
    minVals = dataMat.min(0)
    maxVals = dataMat.max(0)
    rangeVals = maxVals - minVals

    rows = dataMat.shape[0]
    minMat = np.tile(minVals, (rows, 1))
    maxMat = np.tile(rangeVals, (rows, 1))
    normMat = (dataMat - minMat) / maxMat
    return normMat

def plotxyscatter(axs, xdata, ydata, labelColors, xlabel, ylabel, font, legend):
    axs.scatter(x=xdata, y=ydata, color=labelColors, s=15, alpha=0.5)
    axs_title  = axs.set_title(xlabel+u'与'+ylabel, FontProperties=font)
    axs_xlabel = axs.set_xlabel(xlabel, FontProperties=font)
    axs_ylabel = axs.set_ylabel(ylabel, FontProperties=font)
    plt.setp(axs_title, size=9, weight='bold', color='red')
    plt.setp(axs_xlabel, size=7, weight='bold', color='black')
    plt.setp(axs_ylabel, size=7, weight='bold', color='black')
    axs.legend(handles=legend)

def showdatas(dataMat, dataLebel):
    font = FontProperties(size=14)

    fig, axs = plt.subplots(nrows=2, ncols=2,
                            sharex=False, sharey=False, figsize=(13,8))
    numberOfLabels = len(dataLabel)
    labelColors = []
    for i in dataLabel:
        if i == 1:
            labelColors.append('black')
        elif i == 2:
            labelColors.append('orange')
        elif i == 3:
            labelColors.append('red')

    didntlike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntlike')
    smalldose = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='smalldoses')
    largedose = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='largedoses')
    legend = [didntlike, smalldose, largedose]

    str1 = u'每年获得的飞行常客里程数'
    str2 = u'玩视频游戏所消耗时间占比'
    str3 = u'每周消费的冰激凌公式数'
    plotxyscatter(axs[0][0], dataMat[:, 0], dataMat[:, 1], labelColors, str1, str2, font, legend)
    plotxyscatter(axs[0][1], dataMat[:, 0], dataMat[:, 2], labelColors, str1, str3, font, legend)
    plotxyscatter(axs[1][0], dataMat[:, 1], dataMat[:, 2], labelColors, str2, str3, font, legend)

    plt.show()

def datingClassTest(k):
    filename = 'datingTestSet.txt'
    datamat, datalabel = file2matrix(filename)

    ratio = 0.1

    normmat = autoNorm(datamat)
    rows = normmat.shape[0]

    testrows = int(rows * ratio)
    errcount = 0
    for i in range(testrows):
        re = classify0(normmat[i, :], normmat[testrows:rows, :], datalabel[testrows:rows], k)
        #print('分类结果：%d\t真实类别：%d' % (re, datalabel[i]))
        if re != datalabel[i]:
            errcount += 1
    print('错误率：%f%%' % (100.0*errcount/testrows))

if __name__ == '__main__':
    #filename = 'datingTestSet.txt'
    #dataMat, dataLabel = file2matrix(filename)
    #showdatas(dataMat, dataLabel)
    #normDataMat = autoNorm(dataMat)

    for i in range(1, 11):
        datingClassTest(i)
