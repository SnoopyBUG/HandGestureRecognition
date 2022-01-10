import torch
from torch import nn
import torchvision
import sys
from PIL import Image
import time
from numpy import array, newaxis
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels1, out_channels1, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(out_channels1, out_channels2, kernel_size=1)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(in_channels, out_channels2, kernel_size=1, stride=stride)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.bn2 = nn.BatchNorm2d(out_channels1)
        self.bn3 = nn.BatchNorm2d(out_channels2)

    def forward(self, X):
        Y = torch.nn.functional.relu(self.bn1(self.conv1(X)))
        Y = torch.nn.functional.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        return torch.nn.functional.relu(Y + X)


def resnet_block(in_channels, out_channels1, out_channels2, num_residduals, stride=1):
    blk = []
    for i in range(num_residduals):
        if i == 0:
            blk.append(Residual(in_channels, out_channels1, out_channels2, use_1x1conv=True, stride=stride))
            in_channels = out_channels2
        else:
            blk.append(Residual(in_channels, out_channels1, out_channels2))
    return nn.Sequential(*blk)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)


class gesture_datasets(torch.utils.data.Dataset):
    def __init__(self, txt_path, transform=None):
        lines = open(txt_path, 'r')
        imgs = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, index):
        image_path = self.imgs[index][0]
        label = self.imgs[index][1]
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)


def load_data_gesture(batch_size, resize=None):
    transform = []
    if resize:
        transform.append(torchvision.transforms.Resize(size=resize))
    transform.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform)
    gesture_train = gesture_datasets('path-label_train.txt', transform=transform)
    t1 = time.time()
    gesture_test = gesture_datasets('path-label_test.txt', transform=transform)
    t2 = time.time()
    t_ImgLoadDelay = t2 - t1
    num_workers = 0 if sys.platform.startswith('win') else 4
    train_iter = torch.utils.data.DataLoader(gesture_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(gesture_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter, t_ImgLoadDelay


def evaluate_accuracy(data_iter, net, device=None, BelieveThreshold=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    net.eval()
    t1 = time.time()
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X.to(device))
            # 如果置信度不高，可以考虑将其分类为“其他手势”
            for row in y_hat:
                if row[10] < row.max() < BelieveThreshold:
                    row[10] = 1
            # 结束
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    t2 = time.time()
    net.train()
    return acc_sum / n, (t_ImgInDelay+t2-t1) / n


def ConfusionMatrix(net, test_iter, device, FigName='ConfusionMatrix.jpg'):
    if isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    net.eval()
    confusion = array([[0]*11]*11)
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X.to(device)).argmax(dim=1)
            for i in range(y.shape[0]):
                confusion[y[i]][y_hat[i]] += 1
    print(confusion)
    confusion = confusion.astype(float) / confusion.sum(axis=1)[:, newaxis]
    plt.imshow(confusion, cmap=plt.cm.binary)
    actions = ['hand_clapping', 'right_hand_wave', 'left_hand_wave', 'right_arm_clockwise', 'right_arm_counter_clockwise', 'left_arm_clockwise', 'left_arm_counter_clockwise', 'arm_roll', 'air_drums', 'air_guitar', 'other_gestures']
    actions_zh = ['鼓掌', '右手挥手', '左手挥手', '右臂顺时针', '右臂逆时针', '左臂顺时针', '左臂逆时针', '绕臂', '空气架子鼓', '空气吉他', '其他手势']
    plt.xticks(range(11), actions_zh, rotation=-45)
    plt.yticks(range(11), actions_zh)
    # 显示colorbar
    plt.colorbar()
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('混淆矩阵')
    # 在图中标注数量/概率信息
    for x in range(11):
        for y in range(11):
            info = '%d' %(round(confusion[y, x], 2) * 100)    # 不是confusion[x, y].图中横坐标是x，纵坐标是y
            if eval(info):
                plt.text(x, y, info+'%', verticalalignment='center', horizontalalignment='center', color='dodgerblue')
    plt.tight_layout()    # 图形显示更加紧凑
    plt.savefig(FigName)
    plt.show()


def TrainAndTest(net, train_iter, test_iter, optimizer, device, num_epochs, scheduler=None, BelieveThreshold=None):
    net = net.to(device)
    print("Training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        train_loss = train_l_sum / batch_count
        if scheduler:
            scheduler.step(train_loss)
        test_acc, test_delay = evaluate_accuracy(test_iter, net, BelieveThreshold=BelieveThreshold)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, test delay %.5f sec, time %.1f sec'
              %(epoch + 1, train_loss, train_acc_sum / n, test_acc, test_delay, time.time() - start))
    # 混淆矩阵
    ConfusionMatrix(net, test_iter, device, FigName='ConfusionMatrix-BT0.55.jpg')


net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
net.add_module('resnet_block1', resnet_block(64, 64, 256, 3))
net.add_module('resnet_block2', resnet_block(256, 128, 512, 4))
net.add_module('resnet_block3', resnet_block(512, 256, 1024, 6))
net.add_module('resnet_block4', resnet_block(1024, 512, 2048, 3))
net.add_module('global_avg_pool', GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(2048, 11)))

net = torch.nn.DataParallel(net)
batch_size = 250
train_iter, test_iter, t_ImgInDelay = load_data_gesture(batch_size)
lr, num_epochs = 0.0001, 250
BelieveThreshold = 0.55
print('lr:%.4f, num_epochs:%d, BelieveThreshold:%.2f' %(lr, num_epochs, BelieveThreshold))
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-8)
TrainAndTest(net, train_iter, test_iter, optimizer, device, num_epochs, scheduler=scheduler, BelieveThreshold=BelieveThreshold)
