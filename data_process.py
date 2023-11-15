import os
import re   # 需要进行去除空格号
import pandas as pd
import random

# 数据集路径
datafiles = "./dataset/comment"
datafiles1 = "./dataset/sentiment_data"

# 读取数据，处理成训练格式
labels = {}
label_keeper=[]

"""for i in os.listdir(datafiles):
    keyword = i.split('.')[0]
    if keyword not in label_keeper:
        label_keeper.append(keyword)
        labels[keyword] = []
    # 读取xlsx文件数据
    print(os.path.join(datafiles, i))
    data = pd.read_excel(os.path.join(datafiles, i), sheet_name='Sheet1', usecols=[3]).values
    for line in data:
        corpus = re.sub('\s+',' ', str(line[0]))
        corpus = re.sub('[#@]+','',corpus)
        if corpus not in labels[keyword]:
            labels[keyword].append(corpus)"""


for i in os.listdir(datafiles1):
    keyword = i.split('.')[0]
    if keyword not in label_keeper:
        label_keeper.append(keyword)
        labels[keyword] = []
    # 读取txt文件数据
    print(os.path.join(datafiles1, i))
    with open(os.path.join(datafiles1, i), 'r', encoding='utf-8') as data:
        for line in data.readlines():
            corpus = re.sub('\s+', ' ', str(line))
            corpus = re.sub('[#@]+', '', corpus)
            if corpus not in labels[keyword]:
                labels[keyword].append(corpus)

other = "./data/other.txt"
other_data = open(other, 'r', encoding='utf-8').readlines()
for i in other_data:
    re.sub('\n', '', i)
    labels["其他"].append(re.sub('\n', '', i))

with open('data/label_mapping.txt', 'w', encoding='utf-8')as lm:
    label_mapping = {}
    l="negative"
    label_mapping[l]=len(label_mapping)
    lm.write(str(len(label_mapping)-1) + ',' + l + '\n')
    l="neutral"
    label_mapping[l]=len(label_mapping)
    lm.write(str(len(label_mapping)-1) + ',' + l + '\n')
    l="positive"
    label_mapping[l]=len(label_mapping)
    lm.write(str(len(label_mapping)-1) + ',' + l + '\n')

with open('data/test.txt', 'w', encoding='utf-8')as ins_test_file:
    with open('data/train.txt', 'w', encoding='utf-8')as ins_file:
        with open('data/dev.txt', 'w', encoding='utf-8')as ins_dev_file:
            for k, v in labels.items():
                split_num = int((len(v)-3)*0.97)
                for h in v[0:3]:
                    ins_test_file.write(str(label_mapping[k]) + '\t' + h + '\n')
                for d in v[4:split_num]:
                    ins_file.write(str(label_mapping[k]) + '\t' + d + '\n')
                for t in v[split_num:]:
                    ins_dev_file.write(str(label_mapping[k]) + '\t' + t + '\n')


t_data = open('data/train.txt', 'r', encoding='utf-8').readlines()
random.shuffle(t_data)
with open('data/train.txt', 'w', encoding='utf-8')as ins_file:
    for i in t_data:
        ins_file.write(i)