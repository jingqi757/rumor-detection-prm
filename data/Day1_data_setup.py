"""Day1: 加载本地数据集并统一格式"""
from datasets import load_from_disk, Dataset, DatasetDict
import os
import hashlib
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 方案A：AVeriTeC（推荐，天然多跳证据链）
# ============================================================
averitec = load_from_disk(os.path.join(DATA_DIR, "AVeriTeC"))
averitec_test = load_from_disk(os.path.join(DATA_DIR, "AVeriTeC_test"))
print("=== AVeriTeC ===")
print(f"train: {len(averitec['train'])}, dev: {len(averitec['dev'])}, test: {len(averitec_test)}")

sample = averitec['train'][0]
print(f"claim: {sample['claim'][:100]}...")
print(f"label: {sample['label']}")
print(f"questions (first): {sample['questions'][0]['question'][:100]}...")

# ============================================================
# 方案B：LIAR-PLUS（备选，有justification文本）
# ============================================================
# TSV 列: id, json_id, label, statement, subject, speaker, job_title,
#          state, party, barely_true, false, half_true, mostly_true,
#          pants_on_fire, context, justification

def load_liar_plus_tsv(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split('\t')
            data.append({
                'id': fields[0] if len(fields) > 0 else '',
                'json_id': fields[1] if len(fields) > 1 else '',
                'label': fields[2] if len(fields) > 2 else '',
                'claim': fields[3] if len(fields) > 3 else '',
                'subject': fields[4] if len(fields) > 4 else '',
                'speaker': fields[5] if len(fields) > 5 else '',
                'speaker_job': fields[6] if len(fields) > 6 else '',
                'state': fields[7] if len(fields) > 7 else '',
                'party': fields[8] if len(fields) > 8 else '',
                'barely_true_count': fields[9] if len(fields) > 9 else '',
                'false_count': fields[10] if len(fields) > 10 else '',
                'half_true_count': fields[11] if len(fields) > 11 else '',
                'mostly_true_count': fields[12] if len(fields) > 12 else '',
                'pants_on_fire_count': fields[13] if len(fields) > 13 else '',
                'context': fields[14] if len(fields) > 14 else '',
                'justification': fields[15] if len(fields) > 15 else '',
            })
    return data

tsv_dir = os.path.join(DATA_DIR, "LIAR-PLUS/dataset/tsv")
liar_train = load_liar_plus_tsv(os.path.join(tsv_dir, "train2.tsv"))
liar_val   = load_liar_plus_tsv(os.path.join(tsv_dir, "val2.tsv"))
liar_test  = load_liar_plus_tsv(os.path.join(tsv_dir, "test2.tsv"))
print(f"\n=== LIAR-PLUS ===")
print(f"train: {len(liar_train)}, val: {len(liar_val)}, test: {len(liar_test)}")
print(f"labels: {set(x['label'] for x in liar_train[:500])}")
print(f"claim: {liar_train[0]['claim'][:100]}...")
print(f"justification: {liar_train[0]['justification'][:100]}...")

# ============================================================
# 统一格式
# ============================================================
AVERITEC_MAP = {
    'Supported': 'Supported',
    'Refuted': 'Refuted',
    'Conflicting Evidence/Cherrypicking': 'NEI',
    'Not Enough Evidence': 'NEI',
}
def unify_averitec(sample, split_name):
    claim_hash = hashlib.md5(sample['claim'].encode()).hexdigest()[:8]
    return {
        'id': f"averitec_{split_name}_{claim_hash}",
        'claim': sample['claim'],
        'label': AVERITEC_MAP.get(sample['label'], 'NEI'),
        'evidence_chain': [
            {
                'question': q.get('question', ''),
                'answers': q.get('answers', [])   # 答案才是证据内容
            }
            for q in sample.get('questions', [])
        ],
        'justification': sample.get('justification', ''),
        'speaker': sample.get('speaker', ''),
        'source': 'averitec',
    }

def unify_averitec_test(sample):
    return {
        'id': f"averitec_test_{sample.get('id', sample.get('claim_id', 'unknown'))}",
        'claim': sample['claim'],
        'label': None,  # 测试集无标签
        'evidence_chain': [
            {
                'question': q.get('question', ''),
                'answers': q.get('answers', [])   # 答案才是证据内容
            }
            for q in sample.get('questions', [])
        ],
        'justification': '',
        'speaker': sample.get('speaker', ''),
        'source': 'averitec',
    }
LIAR_BINARY = {
    'true': 'Supported',
    'mostly-true': 'Supported',
    'half-true': 'NEI',
    'barely-true': 'NEI',
    'false': 'Refuted',
    'pants-fire': 'Refuted',
}
def unify_liar(sample, split_name):
    return {
        'id': f"liar_{split_name}_{sample['id']}",
        'claim': sample['claim'],
        'label': LIAR_BINARY.get(sample['label'], 'NEI'),
        'label_original': sample['label'],  # 保留原始6类标签备用
        'evidence_chain': [],
        'justification': sample['justification'],
        'speaker': sample['speaker'],
        'source': 'liar',
    }
def validate_unified(ds, name, n=3):
    print(f"\n--- {name} 验证 ---")
    print(f"总数: {len(ds)}")
    print(f"字段: {list(ds[0].keys())}")
    for i in range(min(n, len(ds))):
        s = ds[i]
        assert s['claim'], f"样本{i} claim 为空"
        assert s['label'] is not None or name == 'test', f"样本{i} label 为空"
        ec = s['evidence_chain']
        print(f"  [{i}] label={s['label']}, evidence_chain长度={len(ec)}")
        if ec:
            print(f"       第一条证据: {str(ec[0])[:80]}...")
    print("✓ 验证通过")



averitec_unified = DatasetDict({
    'train': Dataset.from_list([unify_averitec(x, 'train') for x in averitec['train']]),
    'dev':   Dataset.from_list([unify_averitec(x, 'dev') for x in averitec['dev']]),
})
averitec_unified_test = Dataset.from_list([
    unify_averitec_test(x) for x in averitec_test
])

liar_unified = DatasetDict({
    'train': Dataset.from_list([unify_liar(x, 'train') for x in liar_train]),
    'val':   Dataset.from_list([unify_liar(x, 'val') for x in liar_val]),
    'test':  Dataset.from_list([unify_liar(x, 'test') for x in liar_test]),
})
validate_unified(averitec_unified['train'], 'averitec_train')
validate_unified(liar_unified['train'], 'liar_train')
# ============================================================
# 保存
# ============================================================
out_dir = os.path.join(DATA_DIR, "unified")
os.makedirs(out_dir, exist_ok=True)

averitec_unified.save_to_disk(os.path.join(out_dir, "averitec_unified"))
averitec_unified_test.to_json(os.path.join(out_dir, "averitec_test.jsonl"))
liar_unified.save_to_disk(os.path.join(out_dir, "liar_unified"))

print(f"\n=== 已保存到 {out_dir}/ ===")

# 标签分布
for name, ds in [('averitec_train', averitec_unified['train']),
                  ('averitec_dev', averitec_unified['dev'])]:
    labels = {}
    for x in ds:
        labels[x['label']] = labels.get(x['label'], 0) + 1
    print(f"AVeriTeC {name}: {labels}")

for name, ds in [('liar_train', liar_unified['train']),
                  ('liar_val', liar_unified['val']),
                  ('liar_test', liar_unified['test'])]:
    labels = {}
    for x in ds:
        labels[x['label']] = labels.get(x['label'], 0) + 1
    print(f"LIAR-PLUS {name}: {labels}")
s = averitec_unified['train'][0]
print(s['evidence_chain'][0])  # 看完整结构，确认 answers 是否存在