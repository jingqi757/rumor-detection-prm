# Day2_bestofN_sampling.py
# 环境要求：pip install transformers bitsandbytes accelerate

import torch
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. 加载模型（4-bit量化，8GB显存安全运行）
# ============================================================
MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,  
)

model.eval()
print(f"模型加载完成，显存占用：{torch.cuda.memory_allocated()/1024**3:.1f} GB")

# ============================================================
# 2. Prompt 设计（利用 AVeriTeC 的 answers 字段）
# ============================================================
def build_prompt(claim, evidence_chain):
    """
    evidence_chain 是 Day1 存的结构：
    [{'question': ..., 'answers': [{'answer':..., 'boolean_explanation':...}]}]
    """
    evidence_text = ""
    for i, eq in enumerate(evidence_chain[:3]):   # 最多用3条，控制长度
        q = eq.get('question', '')
        answers = eq.get('answers', [])
        if answers:
            a = answers[0]
            ans_str = a.get('answer', '')
            expl = a.get('boolean_explanation', '')
            evidence_text += f"  Q{i+1}: {q}\n  A{i+1}: {ans_str}. {expl}\n\n"

    if evidence_text:
        evidence_block = f"""Reference Evidence:
{evidence_text}"""
    else:
        evidence_block = "Reference Evidence: None available.\n"

    prompt = f"""You are a professional fact-checker. Verify the following claim step by step.
Use the reference evidence provided. Each step must start with [Step X].

Claim: {claim}

{evidence_block}
Please verify following these steps:
[Step 1] Identify the core assertion and key entities in the claim
[Step 2] Assess what the reference evidence says about the claim
[Step 3] Evaluate consistency between the evidence and the claim
[Step 4] Final verdict

Final Verdict: [Supported/Refuted/NEI]"""
    return prompt


# ============================================================
# 3. Best-of-N 生成
# ============================================================
def generate_chains(claim, evidence_chain, n=5):
    prompt = build_prompt(claim, evidence_chain)

    messages = [
        {"role": "system", "content": "You are a precise fact-checker. Always follow the step format exactly."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True 
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[1]

    # 8GB显存下安全参数
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,       # think需要800 
            do_sample=True,
            temperature=0.6,          #官方推荐
            top_p=0.92,
            top_k=20,
            num_return_sequences=n,   # 一次生成N条（比循环快）
            pad_token_id=tokenizer.eos_token_id,
        )

    chains = []
    for i in range(n):
        generated = outputs[i][input_len:]
        text_out = tokenizer.decode(generated, skip_special_tokens=True)
        chains.append(text_out)

    return chains, prompt


# ============================================================
# 4. 推理链解析
# ============================================================
def parse_chain_qwen3(chain_text):
    """
    兼容模型实际输出的混合格式：
    - <think>...</think> 块
    - [Step1] / [Step 1] / Step 1: / Step1: 等变体
    """
    # ① 提取 think 块内容
    think_match = re.search(r'<think>(.*?)</think>', chain_text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        final_output = chain_text[think_match.end():].strip()
    else:
        # 没有 think 标签，整体都是推理内容
        think_content = chain_text
        final_output = chain_text

    # ② 兼容多种 Step 格式的正则
    # 覆盖：[Step1] [Step 1] Step1: Step 1: **Step 1**
    step_pattern = re.compile(
        r'(?:\[Step\s*(\d+)\]|Step\s*(\d+)\s*[:：]|\*\*Step\s*(\d+)\*\*)'
        r'(.*?)(?=(?:\[Step\s*\d+\]|Step\s*\d+\s*[:：]|\*\*Step\s*\d+\*\*)|'
        r'(?:Final\s*Verdict|Step\s*4\s*[:：].*?(?:Supported|Refuted|NEI))|$)',
        re.DOTALL | re.IGNORECASE
    )

    steps = []
    # 先在 think 块里找
    for m in step_pattern.finditer(think_content):
        step_num = m.group(1) or m.group(2) or m.group(3)
        content = m.group(4).strip()
        if content and len(content) > 10:  # 过滤空步骤
            steps.append({
                'step_id': int(step_num),
                'content': content
            })

    # think 块里没找到，再在完整输出里找（模型有时不用think块）
    if not steps:
        for m in step_pattern.finditer(chain_text):
            step_num = m.group(1) or m.group(2) or m.group(3)
            content = m.group(4).strip()
            if content and len(content) > 10:
                steps.append({
                    'step_id': int(step_num),
                    'content': content
                })

    # ③ 提取最终判断（覆盖大小写和多种表达）
    verdict_pattern = re.compile(
        r'(?:Final\s*Verdict|verdict)[^\w]*(Supported|Refuted|NEI|'
        r'Not\s*Enough\s*Evidence|Conflicting)',
        re.IGNORECASE
    )
    # 优先从 final_output 找，再从完整文本找
    vm = verdict_pattern.search(final_output) or verdict_pattern.search(chain_text)
    
    if vm:
        raw_verdict = vm.group(1).strip().lower()
    else:
        # 兜底：在最后200字符里找关键词
        tail = chain_text[-200:].lower()
        if 'refuted' in tail:
            raw_verdict = 'refuted'
        elif 'supported' in tail:
            raw_verdict = 'supported'
        else:
            raw_verdict = 'unknown'

    # ④ 统一 verdict
    verdict_map = {
        'supported': 'Supported',
        'refuted': 'Refuted',
        'nei': 'NEI',
        'not enough evidence': 'NEI',
        'conflicting': 'NEI',
        'unknown': 'Unknown',
    }
    verdict = verdict_map.get(raw_verdict, 'Unknown')

    return steps, verdict, think_content

# ============================================================
# 5. 判断推理链正确性（ORM 弱监督信号）
# ============================================================

MAP = {
        'supported': ['supported', 'true'],
        'refuted':   ['refuted', 'false'],
        'nei':       ['nei', 'not enough evidence',
                      'conflicting evidence/cherrypicking',
                      'conflicting evidence', 'unknown'],
    }

def is_verdict_correct(predicted, gold_label):
    predicted = predicted.strip().lower()
    gold_label = gold_label.strip().lower()
    for canonical, variants in MAP.items():
        if gold_label in variants or gold_label == canonical:
            gold_canonical = canonical
            break
    else:
        gold_canonical = gold_label

    pred_canonical = predicted
    for canonical, variants in MAP.items():
        if predicted in variants:
            pred_canonical = canonical
            break

    return pred_canonical == gold_canonical


# ============================================================
# 6. 主流程
# ============================================================
def process_sample(sample, n=5):
    claim          = sample['claim']
    gold_label     = sample['label']
    evidence_chain = sample['evidence_chain']

    chains_raw, prompt_used = generate_chains(claim, evidence_chain, n=n)
    results = []
    for i, chain in enumerate(chains_raw):
        steps, verdict, think_content = parse_chain_qwen3(chain)
        correct = is_verdict_correct(verdict, gold_label)

        results.append({
            'chain_id':          i,
            'raw_chain':         chain,
            'steps':             steps,
            'predicted_verdict': verdict,
            'gold_label':        gold_label,
            'is_correct':        correct,
            'num_steps_parsed':  len(steps),
        })

    return results, prompt_used


# ============================================================
# 7. 批量处理 + 保存（先跑前50条调试）
# ============================================================
def run_batch(dataset, n_samples=50, n_chains=5, out_path="day2_results.jsonl"):
    all_results = []
    correct_counts = []
    parse_fail_counts = []

    for idx in range(min(n_samples, len(dataset))):
        sample = dataset[idx]
        print(f"[{idx+1}/{n_samples}] claim: {sample['claim'][:60]}...")

        try:
            results, _ = process_sample(sample, n=n_chains)

            # 统计
            n_correct = sum(r['is_correct'] for r in results)
            n_parse_fail = sum(r['num_steps_parsed'] == 0 for r in results)
            correct_counts.append(n_correct)
            parse_fail_counts.append(n_parse_fail)

            all_results.append({
                'sample_id':    sample['id'],
                'claim':        sample['claim'],
                'gold_label':   sample['label'],
                'chains':       results,
            })

            print(f"  正确链: {n_correct}/{n_chains}, "
                  f"解析失败: {n_parse_fail}/{n_chains}, "
                  f"verdicts: {[r['predicted_verdict'] for r in results]}")

        except Exception as e:
            print(f"  ⚠️  处理失败: {e}")
            continue

        # 每10条存一次，防止崩掉丢数据
        if (idx + 1) % 10 == 0:
            _save_jsonl(all_results, out_path)
            print(f"  已保存 {len(all_results)} 条到 {out_path}")

    _save_jsonl(all_results, out_path)

    # 汇总统计
    import numpy as np
    print("\n" + "="*50)
    print("Day 2 汇总统计")
    print("="*50)
    print(f"处理样本数: {len(all_results)}")
    print(f"平均正确链数: {np.mean(correct_counts):.2f} / {n_chains}")
    print(f"正确率 > 0: {sum(c > 0 for c in correct_counts) / len(correct_counts):.1%}")
    print(f"全部正确:   {sum(c == n_chains for c in correct_counts) / len(correct_counts):.1%}")
    print(f"全部错误:   {sum(c == 0 for c in correct_counts) / len(correct_counts):.1%}")
    print(f"平均解析失败: {np.mean(parse_fail_counts):.2f} / {n_chains}")
    print(f"\n→ 关键数字：正确链比例分布决定 Day3 标注策略")

    return all_results


def _save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    unified_dir = os.path.join(DATA_DIR, "unified")
    averitec = load_from_disk(os.path.join(unified_dir, "averitec_unified"))
    train_ds = averitec['train']
    
    out_path = os.path.join(DATA_DIR, "day2_results.jsonl")

    # 第一步：先跑5条看效果
    print("=== 快速验证（5条）===")
    """for i in range(5):
        results, prompt = process_sample(train_ds[i], n=3)
        print(f"\n--- 样本 {i} ---")
        print(f"claim: {train_ds[i]['claim'][:80]}")
        print(f"gold:  {train_ds[i]['label']}")
        for r in results:
            print(f"  chain{r['chain_id']}: verdict={r['predicted_verdict']}, "
                  f"correct={r['is_correct']}, steps={r['num_steps_parsed']}")
"""
    # 确认格式OK后取消注释跑完整50条
    run_batch(train_ds, n_samples=50, n_chains=5, out_path=out_path)
