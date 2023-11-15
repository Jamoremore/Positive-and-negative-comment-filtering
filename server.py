import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request

# 模型加载参数
device = 'cpu'  # 可指定GPU设备
saved_model_path = 'E:\Codes\Duomai\mon4\\task2\data\checkpoints\model_best'
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
model.to(device).eval()

def inference(sentences, batch_size=32, max_seq_len=300):
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
        output = torch.argmax(output, dim=-1).cpu().tolist()
        res.extend(output)
    opt=[]
    for i in res:
        if i == 0:
            opt.append('消极')
        elif i == 1:
            opt.append('中立')
        elif i == 2:
            opt.append('积极')
    # print(opt)
    return opt

if __name__ == '__main__':
    inference("这是坏东西")
    inference("这是好东西")
    inference("这是什么东西")
    for i in range(10000):
        print(str(i)+str(inference("这是坏东西")))
    # app.run(debug=True, host='0.0.0.0', port=5000)


