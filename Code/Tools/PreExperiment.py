file = open('深度学习/大创/IBMDvsGesture/IBM-db/train/path-label_train.txt', 'r')
lines = file.readlines()
fpre = open('深度学习/大创/IBMDvsGesture/IBM-bd/train/path-label_train-pre.txt', 'w')
i = 1
for line in lines:
    if i % 10 == 0:
        fpre.write(line)
    i += 1
file.close()
fpre.close()
