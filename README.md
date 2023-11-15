# 正负评论分析

采用'bert-base-chinese'预训练模型做评论分类，分为0,negative;1,neutral;2,positive;三个类别。

## 文件结构

| 文件名                 | 功能     |
| :--------------------- | -------- |
| **train.py**           | 模型训练 |
| **data_process.py**    | 生成数据 |
| **inference.py**       | 快速测试 |
| **iTrainingLogger.py** | 绘图     |

## 运行
```
python train.py
```
