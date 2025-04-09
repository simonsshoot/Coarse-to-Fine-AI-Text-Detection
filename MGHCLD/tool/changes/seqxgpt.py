import json
import csv
input_file = 'MGHCLD/detect_data/sexgpt_origin/eval.jsonl'
output_file = 'MGHCLD/detect_data/sexgpt_origin/valid.csv'

# 打开输入和输出文件
with open(input_file, 'r', encoding='utf-8-sig') as infile, open(output_file, 'w', newline='', encoding='utf-8-sig') as outfile:
    reader = infile.readlines()
    writer = csv.writer(outfile)
    
    writer.writerow(['Generation', 'label'])
    for line in reader:
        data = json.loads(line.strip())
        
        generation = data['text']
        label = 'machine_origin'if data['labels']==1 else 'human'
        writer.writerow([generation, label])

print("changes done:", output_file)
