import numpy as np

def createDataSet():
    group = np.array([[1,101], [5,89], [108,5], [115,8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']

    return group, labels

if __name__ == '__main__':
    group, labels = createDataSet()
    print(group)
    print(labels)
