
from transformers import AutoTokenizer, AutoModel
"""device="cpu"
tokenizer = AutoTokenizer.from_pretrained("E:\\temp\\chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("E:\\temp\\chatglm-6b-int4", trust_remote_code=True).float()
model = model.eval()"""


tokenizer = AutoTokenizer.from_pretrained("E:\\temp\\chatglm-6b-int4", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half()
# 加载量化模型 gpu
# model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
# cpu
model = AutoModel.from_pretrained("E:\\temp\\chatglm-6b-int4", trust_remote_code=True).float()
model = model.eval()