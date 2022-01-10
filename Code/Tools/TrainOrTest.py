file = open('深度学习/大创/IBMDvsGesture/预实验/path-label.txt', 'r')
lines = file.readlines()
ftrain = open('深度学习/大创/IBMDvsGesture/预实验/path-label_train.txt', 'w')
ftest = open('深度学习/大创/IBMDvsGesture/预实验/path-label_test.txt', 'w')
i = 1
for line in lines:
    if i % 5 != 0:
        ftrain.write(line)
    else:
        ftest.write(line)
    i += 1
file.close()
ftrain.close()
ftest.close()