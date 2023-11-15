# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

测试训练好的模型。

Author: pankeyu
Date: 2023/01/11
"""
from typing import List
import time
import torch
from rich import print
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch.nn as nn

# 信息熵
def calc_ent(prob_matrix):
    ents = []
    for prob_list in prob_matrix:
        ent = 0.0
        l = len(prob_list)
        for prob in prob_list:
            ent -= prob*np.log2(prob)
        ents.append(ent/l)
    return ents
class pred(nn.Module):
    def __init__(self,
                 model,
                 tokenizer,
                 device: str,
                 lm_dict,
                 batch_size=16,
                 max_seq_len=128):
        super(pred, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lm_dict = lm_dict
        self.max_seq_len = max_seq_len

    def forward(
            self,
            text
    ):
        ipnuts = tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        res = []
        output = model(
            input_ids=ipnuts['input_ids'].to(self.device),
            token_type_ids=ipnuts['token_type_ids'].to(self.device),
            attention_mask=ipnuts['attention_mask'].to(self.device),
        ).logits
        # 先判断是概率混淆度，如果出现混淆的情况，则判定为‘其他’类
        prob = torch.softmax(output, dim=-1).cpu().tolist()
        output = torch.argmax(output, dim=-1).cpu().tolist()
        res.extend(output)
        print(lm_dict.get(str(res[0])))
        return res

def inference(
    model, 
    tokenizer, 
    sentences: List[str],
    device: str,
    batch_size=16,
    max_seq_len=128,
    ) -> List[int]:
    """
    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        sentences (List[str]): _description_
        batch_size (int, optional): _description_. Defaults to 16.
        max_seq_len (int, optional): _description_. Defaults to 128.

    Returns:
        List[int]: [laebl1, label2, label3, ...]
    """
    res = []
    for i in range(0, len(sentences), batch_size):
        batch_sentence = sentences[i:i+batch_size]
        ipnuts = tokenizer(
            batch_sentence,
            truncation=True,
            max_length=max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        output = model(
            input_ids=ipnuts['input_ids'].to(device),
            token_type_ids=ipnuts['token_type_ids'].to(device),
            attention_mask=ipnuts['attention_mask'].to(device),
        ).logits
        # 先判断是概率混淆度，如果出现混淆的情况，则判定为‘其他’类
        prob = torch.softmax(output, dim=-1).cpu().tolist()
        print(prob)
        # output = torch.argmax(output, dim=-1).cpu().tolist()
        # for index, p in enumerate(prob):
        #     if p[output[index]] > 0.4:
        #         res.append(output[index])
        #     else:
        #         res.append(-1)
        output = torch.argmax(output, dim=-1).cpu().tolist()
        res.extend(output)
    return res


if __name__ == '__main__':
    load_time = time.time()
    # device = 'cpu'                                                  # 指定GPU设备
    device = 'cuda'  # 指定GPU设备
    saved_model_path = 'data/checkpoints/model_best'  # 训练模型存放地址
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path) 
    model.to(device).eval()

    test_file = 'data/test.txt'
    test = [i for i in open(test_file, 'r', encoding='utf-8').readlines()]
    print(time.time()-load_time)
    map_dict = './data/label_mapping.txt'
    lab_map_dict = [i.strip().split(',') for i in open(map_dict, 'r', encoding='utf-8')]
    lm_dict = {}
    for i in lab_map_dict:
        lm_dict[i[0]] = i[-1]

    strat_time = time.time()
    res = inference(model, tokenizer, test, device)
    for i in res:
        print(lm_dict.get(str(i)))
    print('test time：', time.time() - strat_time)

    ans = []
    with open('data/test.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            s = int(items[0])
            ans.append(s)
    accuracy = sum([1 for a, r in zip(ans, res) if a == r])/len(ans)
    print('accuracy:', accuracy)
    pred=pred(model, tokenizer, device, lm_dict, batch_size=64, max_seq_len=120)
    pred("shoe")

