import json
import csv
input_file = '/data/hc3/hc3_train.jsonl'
output_file = 'MGHCLD/detect_data/HC3_origin/train.csv'


with open(input_file, 'r', encoding='utf-8-sig') as infile, open(output_file, 'w', newline='', encoding='utf-8-sig') as outfile:
    reader = infile.readlines()
    writer = csv.writer(outfile)
    

    writer.writerow(['Generation', 'label'])
    for line in reader:
        data = json.loads(line.strip())

        generation = data['article']
        label = data['label'] if data['label']=='human'else 'machine_origin'
        writer.writerow([generation, label])

print("changes done:", output_file)
