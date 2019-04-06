from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import pydotplus

def rawDataToPandasData():
    with open('lenses.txt', 'r') as fr:
        lenses = [lense.strip().split('\t') for lense in fr.readlines()]
    lenses_target = []
    for lense in lenses:
        lenses_target.append(lense[-1])
    lenseslabels = ['age', 'prescipt', 'astigmatic', 'tearrate']
    lenses_dict = {}
    for index in range(len(lenseslabels)):
        lenses_list = []
        for lense in lenses:
            lenses_list.append(lense[index])
        lenses_dict[lenseslabels[index]] = lenses_list

    #print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
    #print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    #print(lenses_pd)

    return lenses_pd, lenses_target


def test():
    lenses_pd, lenses_target = rawDataToPandasData()
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')

    print(clf.predict([[1,1,1,0]]))


if __name__=='__main__':
    test()
