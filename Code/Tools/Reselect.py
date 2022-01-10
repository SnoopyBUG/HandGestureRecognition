import random


def reselect(path, train_path, test_path):
    f_train = open(train_path, 'a')
    f_test = open(test_path, 'a')
    lines = open(path, 'r').readlines()
    l = len(lines)
    test_list = random.sample(range(0, l), l//5)
    print(len(test_list))
    i = 0
    for line in lines:
        if i in test_list:
            f_test.write(line)
        else:
            f_train.write(line)
        i += 1


path1 = 'IBMDvsGesture/PRO/train/path-label_train.txt'
path2 = 'IBMDvsGesture/PRO/test/path-label_test.txt'
train_path = 'path-label_train_max.txt'
test_path = 'path-label_test_max.txt'
reselect(path1, train_path, test_path)
reselect(path2, train_path, test_path)
