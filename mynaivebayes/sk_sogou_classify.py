from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import jieba
import random

def text_processing(folder_path, test_size = 0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    for folder in folder_list:
        filepath = os.path.join(folder_path, folder)
        files = os.listdir(filepath)

        for file in files:
            with open(os.path.join(filepath, file), 'r', encoding='utf-8') as f:
                raw = f.read()
            word_cut = jieba.cut(raw, cut_all=False)
            data_list.append(list(word_cut))
            class_list.append(folder)

    data_class_list = list(zip(data_list, class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True)
    all_words_list, all_words_num = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def stopWordSet(stop_file):
    wordset = set()
    with open(stop_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                wordset.add(line)
    return wordset

def wordfilter(wordlist, topN, stopword = set()):
    feature_words = []
    for i in range(topN, len(wordlist)):
        if len(feature_words) > 1000:
            break   #only need 1000
        word = wordlist[i]
        if not word.isdigit() and word not in stopword and 1<len(word)<5:
            feature_words.append(word)
    return feature_words

def textFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        textset = set(text)
        feature = [1 if word in textset else 0 for word in feature_words]
        return feature
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list

def textClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    folder_path = './SogouC/Sample'
    all_word_list, train_data_list, test_data_list, train_class_list, test_class_list = text_processing(folder_path)
    stop_file = './stopwords_cn.txt'
    stopword = stopWordSet(stop_file)

    test_accuracy_list = []
    topNs = range(0, 1000, 20)
    for topn in topNs:
        feature_words = wordfilter(all_word_list, topn, stopword)
        train_feature_list, test_feature_list = textFeatures(train_data_list, test_data_list, feature_words)
        accuracy = textClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(accuracy)

    print(sum(test_accuracy_list) / len(test_accuracy_list))

    plt.figure()
    plt.plot(topNs, test_accuracy_list)
    plt.title('topNs and test_accuracy')
    plt.xlabel('topNs')
    plt.ylabel('test_accuracy')
    plt.show()
