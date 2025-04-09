import json
import csv
input_file = 'gpt2_checkgptorigintest.jsonl'
output_file = 'MGHCLD/detect_data/watermark/checkgpttemp.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = infile.readlines()
    writer = csv.writer(outfile,delimiter='|')
    

    #writer.writerow(['Generation', 'label'])
    # record=5993
    for line in reader:
        data = json.loads(line.strip())
        
        # id=record
        generation = data['article']
        label = 1 if data['label']=='human'else 0

        writer.writerow([generation, label])
        # record+=1

print("changes done:", output_file)
