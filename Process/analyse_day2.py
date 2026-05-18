# analyze_day2.py
import json

with open("/home/jingqi/research/rumor-detection/prm/data/day2_results.jsonl") as f:
    data = [json.loads(l) for l in f]

total_chains = 0
verdict_counts = {}
unknown_with_steps = 0

for sample in data:
    for chain in sample['chains']:
        v = chain['predicted_verdict']
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        total_chains += 1
        if v == 'Unknown' and chain['num_steps_parsed'] > 0:
            unknown_with_steps += 1

print(f"总链数: {total_chains}")
print(f"verdict分布: {verdict_counts}")
print(f"Unknown中有步骤的: {unknown_with_steps}")
for sample in data:
    for chain in sample['chains']:
        if chain['predicted_verdict'] == 'Unknown' and chain['num_steps_parsed'] > 0:
            print("=== Unknown但有步骤的末尾200字 ===")
            print(chain['raw_chain'][-200:])
            print()
            break
    else:
        continue
    break