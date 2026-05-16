# Day1_data_setup.py
# 安装依赖（消费级GPU友好）
# pip install transformers datasets torch accelerate

from datasets import load_dataset
import json, os
# === 方案A：AVeriTeC（推荐，天然多跳证据链）===
dataset = load_dataset("chenxwh/AVeriTeC")
print(dataset)
# 关键字段：claim, label, questions(证据问题链), answers

# 查看一条样本结构
sample = dataset['train'][0]
print("claim:", sample['claim'])
print("label:", sample['label'])
print("evidence questions:", sample['questions'])

# === 方案B：LIAR-PLUS（备选，有justification文本）===
# 从 https://github.com/Tariq60/LIAR-PLUS 下载
# 关键字段：statement, label, justification

def load_liar_plus(path):
    data = []
    with open(path) as f:
        for line in f:
            fields = line.strip().split('\t')
            data.append({
                'claim': fields[2],
                'label': fields[1],        # true/false/half-true 等6类
                'justification': fields[20] if len(fields) > 20 else ""
            })
    return data

# === 统一格式（两个数据集都转成这个结构）===
def unify_format(sample, source='averitec'):
    if source == 'averitec':
        return {
            'id': sample.get('claim_id', ''),
            'claim': sample['claim'],
            'label': sample['label'],          # "Supported" / "Refuted" / "NEI"
            'evidence_chain': sample.get('questions', []),  # 天然步骤链！
            'source': 'averitec'
        }
    elif source == 'liar':
        return {
            'id': sample.get('id', ''),
            'claim': sample['claim'],
            'label': sample['label'],
            'evidence_chain': [],              # 需要LLM生成
            'justification': sample.get('justification', ''),
            'source': 'liar'
        }