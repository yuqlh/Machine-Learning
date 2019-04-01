from math import log

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']  # 分类属性
    return dataSet, labels

def calcShannonEntroy(dataset):
    classcount = {}
    rowcount = len(dataset)
    for row in dataset:
        classkey = row[-1]
        if classkey not in classcount.keys():
            classcount[classkey] = 0
        classcount[classkey] += 1

    entroy = 0.0
    for key in classcount:
        #print(key, classcount[key])
        prob = float(classcount[key]) / rowcount
        entroy -= prob * log(prob, 2.0)

    return entroy


if __name__ == '__main__':
    ds, labels = createDataSet()
    print(calcShannonEntroy(ds))
