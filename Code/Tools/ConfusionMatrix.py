import matplotlib.pyplot as plt
from numpy import array, newaxis
plt.rcParams['font.sans-serif'] = ['SimHei']

confusion = [[990,    2   , 7 ,   5   , 1  ,  5    ,6 ,  28 , 105 ,  36 ,  21],
             [1 ,1507 ,   0 ,  23 ,  17 ,   0 ,   0 ,   0 ,   1 ,   0 ,   8],
             [0 ,   0 ,1241 ,   0 ,   0 , 181 ,  36 ,   1 ,   0 ,   2 ,  10],
             [10,  106,    1, 1674,  108,    5,    2,    0,   18,    2,  126],
             [1 ,  94 ,   0 , 116 ,1493 ,   5 ,   1 ,   0 ,   7 ,   0 ,  69],
             [0 ,   0 ,   8 ,   0 ,   0 ,1747 ,  87 ,   0 ,   1 ,   0 ,  31],
             [2 ,   0 ,  18 ,   1 ,   0 , 168 ,1533 ,   0 ,   0 ,   1 ,  12],
             [6 ,   0 ,   0 ,   0 ,   0 ,   0 ,   1 ,1628 ,  14 ,   3 ,   4],
             [340,    2,    0,   10,    0,   13,    5,  169, 1051,   15,   48],
             [12 ,   0 ,   2 ,   0 ,   1 ,   4 ,   5 ,  82 ,  75 ,1061 , 358],
             [7  , 41  , 21  , 23  , 13  , 13  , 19  ,  2  , 43  , 36  ,981]]
confusion = array(confusion)
print(confusion)

confusion = confusion.astype(float) / confusion.sum(axis=1)[:, newaxis]
plt.imshow(confusion, cmap=plt.cm.binary)
actions = ['hand_clapping', 'right_hand_wave', 'left_hand_wave', 'right_arm_clockwise', 'right_arm_counter_clockwise', 'left_arm_clockwise', 'left_arm_counter_clockwise', 'arm_roll', 'air_drums', 'air_guitar', 'other_gestures']
actions_zh = ['鼓掌', '右手挥手', '左手挥手', '右臂顺时针', '右臂逆时针', '左臂顺时针', '左臂逆时针', '双臂环绕', '空气架子鼓', '空气吉他', '其他手势']
plt.xticks(range(11), actions_zh, rotation=-45)
plt.yticks(range(11), actions_zh)
# 显示colorbar
plt.colorbar()
plt.xlabel('预测')
plt.ylabel('真实')
plt.title('Confusion Matrix\n混淆矩阵')
# 在图中标注数量/概率信息
Threshold = 0.5
for x in range(11):
    for y in range(11):
        info = '%d' %(round(confusion[y, x], 2) * 100)
        # info = str(round(confusion[y, x], 2) * 100) + '%'    # 不是confusion[x, y].图中横坐标是x，纵坐标是y
        if eval(info):
            color = 'white' if eval(info)>50 else 'black'
            plt.text(x, y, info+'%', verticalalignment='center', horizontalalignment='center', color=color)
plt.tight_layout()    # 图形显示更加紧凑
plt.savefig('Confusion Matrix.jpg')
plt.show()
